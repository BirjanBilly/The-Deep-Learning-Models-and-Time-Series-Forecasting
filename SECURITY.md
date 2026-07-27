# Security and privacy

Do not commit credentials, private cluster addresses, account names, licensed datasets, personal file paths or model artifacts containing confidential data. Use environment variables or a local `.env` file for secrets; `.env` files are ignored by Git.

Before each public release, inspect the staged changes with:

```bash
git diff --cached
```

and scan the repository for likely secrets with a tool such as `gitleaks` or GitHub secret scanning.
