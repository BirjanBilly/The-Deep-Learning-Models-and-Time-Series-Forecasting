from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from riskgraph.data.dataset import Panel
from riskgraph.io import file_sha256
from riskgraph.models.hybrid import TemporalGraphQuantileNet
from riskgraph.models.temporal import TemporalFusionQuantileNet


@dataclass(frozen=True)
class RiskGraphConditionerSpec:
    """Serializable construction metadata for a frozen RiskGraph teacher."""

    model_name: str
    graph_mode: str
    macro_mode: str
    graph_signal_mode: str
    constructor: dict[str, Any]
    target_mean: list[float]
    target_std: list[float]
    quantile_levels: list[float]
    horizons: list[int]
    lookback: int
    source_checkpoint: str
    source_checkpoint_sha256: str


def _variant_modes(checkpoint_path: Path) -> tuple[str, str, str]:
    variant = checkpoint_path.parent.parent.name
    graph_mode = "dynamic"
    macro_mode = "enabled"
    graph_signal_mode = "direction"
    if "graph_static" in variant:
        graph_mode = "static"
    elif "graph_identity" in variant:
        graph_mode = "identity"
    if "macro_disabled" in variant:
        macro_mode = "disabled"
    if "signal_embedding" in variant:
        graph_signal_mode = "embedding"
    return graph_mode, macro_mode, graph_signal_mode


def _read_run_metadata(checkpoint_path: Path) -> dict[str, Any]:
    metadata_path = checkpoint_path.with_name("run_metadata.json")
    if not metadata_path.exists():
        return {}
    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def build_riskgraph_model(constructor: dict[str, Any], model_name: str) -> nn.Module:
    common = {
        "asset_features": int(constructor["asset_features"]),
        "macro_features": int(constructor["macro_features"]),
        "hidden_size": int(constructor["hidden_size"]),
        "lstm_layers": int(constructor["lstm_layers"]),
        "attention_heads": int(constructor["attention_heads"]),
        "dropout": float(constructor["dropout"]),
        "horizons": int(constructor["horizons"]),
        "quantile_levels": [float(value) for value in constructor["quantile_levels"]],
        "target_index": int(constructor["target_index"]),
    }
    if model_name == "temporal":
        return TemporalFusionQuantileNet(**common)
    if model_name == "hybrid":
        return TemporalGraphQuantileNet(
            **common,
            graph_heads=int(constructor["graph_heads"]),
            graph_signal_mode=str(constructor.get("graph_signal_mode", "direction")),
        )
    raise ValueError(f"Unsupported RiskGraph conditioner model: {model_name}")


class RiskGraphConditioner(nn.Module):
    """Frozen direct-forecast model used as a market-state encoder and quantile teacher."""

    def __init__(self, model: nn.Module, spec: RiskGraphConditionerSpec) -> None:
        super().__init__()
        self.model = model
        self.spec = spec
        self.state_dim = int(spec.constructor["hidden_size"])
        self.register_buffer(
            "target_mean",
            torch.tensor(spec.target_mean, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "target_std",
            torch.tensor(spec.target_std, dtype=torch.float32),
            persistent=True,
        )
        self.freeze()

    def freeze(self) -> None:
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):  # type: ignore[override]
        # The teacher remains frozen and deterministic even while the decoder trains.
        super().train(False)
        self.model.eval()
        return self

    def encode(
        self,
        asset: torch.Tensor,
        macro: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.spec.macro_mode == "disabled":
            macro = torch.zeros_like(macro)
        if self.spec.model_name == "temporal":
            assert isinstance(self.model, TemporalFusionQuantileNet)
            state, _, _, _ = self.model.encode_state(asset, macro, adjacency)
        else:
            assert isinstance(self.model, TemporalGraphQuantileNet)
            state, *_ = self.model.encode_state(asset, macro, adjacency)
        scaled_quantiles = self.model.quantile_head(state)
        mean = self.target_mean.view(1, -1, 1).to(
            device=scaled_quantiles.device,
            dtype=scaled_quantiles.dtype,
        )
        std = self.target_std.view(1, -1, 1).to(
            device=scaled_quantiles.device,
            dtype=scaled_quantiles.dtype,
        )
        quantiles = scaled_quantiles * std + mean
        return state, quantiles

    def forward(
        self,
        asset: torch.Tensor,
        macro: torch.Tensor,
        adjacency: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encode(asset, macro, adjacency)

    def export_spec(self) -> dict[str, Any]:
        return asdict(self.spec)

    @classmethod
    def from_spec(cls, value: dict[str, Any]) -> RiskGraphConditioner:
        spec = RiskGraphConditionerSpec(
            model_name=str(value["model_name"]),
            graph_mode=str(value["graph_mode"]),
            macro_mode=str(value["macro_mode"]),
            graph_signal_mode=str(value.get("graph_signal_mode", "direction")),
            constructor=dict(value["constructor"]),
            target_mean=[float(item) for item in value["target_mean"]],
            target_std=[float(item) for item in value["target_std"]],
            quantile_levels=[float(item) for item in value["quantile_levels"]],
            horizons=[int(item) for item in value["horizons"]],
            lookback=int(value["lookback"]),
            source_checkpoint=str(value.get("source_checkpoint", "embedded")),
            source_checkpoint_sha256=str(value.get("source_checkpoint_sha256", "embedded")),
        )
        return cls(build_riskgraph_model(spec.constructor, spec.model_name), spec)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        panel: Panel,
        config: dict[str, Any],
        device: torch.device,
    ) -> RiskGraphConditioner:
        path = Path(checkpoint_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model_name = str(checkpoint["model"])
        metadata = _read_run_metadata(path)
        fallback_graph, fallback_macro, fallback_signal = _variant_modes(path)
        graph_mode = str(metadata.get("graph_mode", fallback_graph))
        macro_mode = str(metadata.get("macro_mode", fallback_macro))
        graph_signal_mode = str(metadata.get("graph_signal_mode", fallback_signal))
        model_config = config["model"]
        constructor: dict[str, Any] = {
            "asset_features": int(panel.asset_features.shape[-1]),
            "macro_features": int(panel.macro_features.shape[-1]),
            "hidden_size": int(model_config["hidden_size"]),
            "lstm_layers": int(model_config["lstm_layers"]),
            "attention_heads": int(model_config["attention_heads"]),
            "dropout": float(model_config["dropout"]),
            "horizons": len(checkpoint["horizons"]),
            "quantile_levels": [float(value) for value in checkpoint["quantiles"]],
            "target_index": int(panel.target_index),
            "graph_heads": int(model_config["graph_heads"]),
            "graph_signal_mode": graph_signal_mode,
        }
        model = build_riskgraph_model(constructor, model_name)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        spec = RiskGraphConditionerSpec(
            model_name=model_name,
            graph_mode=graph_mode,
            macro_mode=macro_mode,
            graph_signal_mode=graph_signal_mode,
            constructor=copy.deepcopy(constructor),
            target_mean=np.asarray(checkpoint["target_mean"], dtype=float).tolist(),
            target_std=np.asarray(checkpoint["target_std"], dtype=float).tolist(),
            quantile_levels=[float(value) for value in checkpoint["quantiles"]],
            horizons=[int(value) for value in checkpoint["horizons"]],
            lookback=int(config["features"]["lookback"]),
            source_checkpoint=str(path),
            source_checkpoint_sha256=file_sha256(path),
        )
        conditioner = cls(model, spec).to(device)
        conditioner.freeze()
        return conditioner


def precompute_conditioner_outputs(
    windows,
    conditioner: RiskGraphConditioner,
    device: torch.device,
    batch_size: int = 256,
):
    """Cache frozen RiskGraph states and teacher quantiles once per origin.

    The raw histories remain attached for auditability, while repeated Tail-GAN
    epochs use the cached arrays and avoid rerunning the frozen teacher.
    """

    from dataclasses import replace

    if windows.state_asset is None:
        raise ValueError("Conditioning histories must be attached before precomputation")
    assert windows.state_macro is not None
    assert windows.state_adjacency is not None
    states: list[np.ndarray] = []
    quantiles: list[np.ndarray] = []
    conditioner = conditioner.to(device)
    conditioner.freeze()
    with torch.no_grad():
        for start in range(0, len(windows.origins), int(batch_size)):
            end = min(start + int(batch_size), len(windows.origins))
            state, teacher = conditioner(
                torch.from_numpy(windows.state_asset[start:end]).to(device),
                torch.from_numpy(windows.state_macro[start:end]).to(device),
                torch.from_numpy(windows.state_adjacency[start:end]).to(device),
            )
            states.append(state.float().cpu().numpy())
            quantiles.append(teacher.float().cpu().numpy())
    return replace(
        windows,
        state_embedding=np.concatenate(states, axis=0).astype(np.float32),
        teacher_quantiles=np.concatenate(quantiles, axis=0).astype(np.float32),
    )
