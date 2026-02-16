Basic install (core deps: torch, transformers, einops, faiss-cpu):
`pip install -e .`

With optional extras:
`pip install -e ".[gpu]"       # faiss-gpu instead of cpu`
`pip install -e ".[logging]"   # adds wandb`
`pip install -e ".[dev]"       # adds pytest, pytest-cov`
`pip install -e ".[gpu,logging,dev]"  # all extras`

The -e flag installs in editable/development mode so code changes take effect immediately without reinstalling.       

Dependencies are defined in pyproject.toml:10-15 (core) and pyproject.toml:17-20 (optional)
