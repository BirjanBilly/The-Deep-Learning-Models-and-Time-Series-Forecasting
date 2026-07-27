from __future__ import annotations


def llmtime_variant_name(
    token_mode: str,
    use_side_info: bool,
    riskgraph_variant: str | None,
    suffix: str | None,
) -> str:
    """Return a stable output-directory name for an LLMTIME experiment variant."""
    parts = ["llmtime", token_mode]
    if use_side_info:
        parts.append("side")
    if riskgraph_variant:
        parts.extend(["rg", riskgraph_variant.replace("__", "_")])
    if suffix:
        parts.append(suffix)
    return "__".join(parts)
