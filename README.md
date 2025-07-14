```bash
# Install UV
pip install uv

# Sync the environment
uv sync --locked --no-install-project --no-dev

# Run the training script
uv run scripts/train.py --config configs/default.yaml
```
