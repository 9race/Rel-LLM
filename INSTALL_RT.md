# Installation Guide: Rel-LLM + Relational Transformer

This guide installs both Rel-LLM and Relational Transformer dependencies in a single conda environment.

## Prerequisites

- Conda environment with Python 3.12 (already created)
- CUDA-capable GPU (for PyTorch)

## Step 1: Activate Environment

```bash
conda activate llm  # or your environment name
```

## Step 2: Install PyTorch and Core Dependencies

```bash
# Install PyTorch (RT requires 2.6.0, but Rel-LLM may work with 2.5.0)
# Using RT's requirement (2.6.0) - check CUDA version compatibility
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Core dependencies for Rel-LLM
pip install wandb pandas pillow pyarrow pooch
pip install relbench[full]
pip install torch-frame[full]
pip install transformers peft

# PyG dependencies (install after PyTorch)
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html
```

## Step 3: Install Relational Transformer Dependencies

```bash
# RT-specific Python packages
pip install sentence-transformers einops strictfire ml-dtypes orjson polars
pip install scikit-learn  # Note: RT requires <1.6.0, but newer versions may work
pip install maturin-import-hook maturin[patchelf]
pip install matplotlib ipykernel

# Optional: wandb (already installed above)
```

## Step 4: Install Rust (for RT's sampler)

```bash
# Install Rust compiler
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Verify installation
rustc --version
```

## Step 5: Build RT's Rust Sampler

```bash
cd /dfs/user/graceluo/Rel-LLM/relational-transformer/rustler

# Build the Rust module
maturin develop --release

# Verify build
cd /dfs/user/graceluo/Rel-LLM
python -c "from rt_adapter import create_rt_loader; print('✓ RT adapter loaded successfully!')"
```

## Step 6: Verify Installation

```bash
# Test Rel-LLM imports
python -c "import torch; import relbench; import torch_frame; print('✓ Rel-LLM dependencies OK')"

# Test RT imports
python -c "from rt_adapter import create_rt_loader; print('✓ RT dependencies OK')"
```

## Troubleshooting

### If PyTorch 2.6.0 is not available for your CUDA version:
- Check available versions: https://pytorch.org/get-started/previous-versions/
- RT may work with 2.5.0, but 2.6.0 is recommended

### If maturin build fails:
- Ensure Rust is installed: `rustc --version`
- Try: `maturin develop --release --no-sdist`

### If scikit-learn version conflict:
- RT requires `<1.6.0`, but newer versions may work
- If issues: `pip install 'scikit-learn<1.6.0'`

### If PyG packages fail:
- Check PyTorch version matches: `python -c "import torch; print(torch.__version__)"`
- Use correct CUDA version in PyG URL

## Quick Install Script

Save this as `install_all.sh`:

```bash
#!/bin/bash
set -e

conda activate llm

echo "Installing PyTorch..."
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "Installing Rel-LLM dependencies..."
pip install wandb pandas pillow pyarrow pooch
pip install relbench[full]
pip install torch-frame[full]
pip install transformers peft
pip install pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv \
  -f https://data.pyg.org/whl/torch-2.6.0+cu124.html

echo "Installing RT dependencies..."
pip install sentence-transformers einops strictfire ml-dtypes orjson polars
pip install scikit-learn maturin-import-hook maturin[patchelf] matplotlib ipykernel

echo "Building RT Rust sampler..."
cd relational-transformer/rustler
maturin develop --release
cd ../..

echo "Verifying installation..."
python -c "from rt_adapter import create_rt_loader; print('✓ All dependencies installed!')"
```

Make it executable and run:
```bash
chmod +x install_all.sh
./install_all.sh
```




