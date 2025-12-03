#!/bin/bash
# Install RT dependencies in current environment
# These are the key dependencies from relational-transformer/pyproject.toml

pip install ml_dtypes
pip install sentence-transformers
pip install wandb
pip install einops
pip install strictfire
pip install orjson
pip install scikit-learn
pip install polars
pip install matplotlib

echo "RT dependencies installed!"




