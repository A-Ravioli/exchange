#!/bin/bash
# run all three training experiments in parallel

echo "🚀 launching all training experiments..."
echo "this will run for DAYS. go live your life."
echo ""

# create checkpoint directories
mkdir -p checkpoints/rl_selfplay checkpoints/evolution checkpoints/hybrid

# activate venv if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# check if wandb is installed
python -c "import wandb" 2>/dev/null || {
    echo "installing wandb..."
    uv pip install wandb
}

# login to wandb if needed
wandb login

# launch three training runs in separate terminals
echo "launching 1/3: RL self-play..."
osascript -e 'tell app "Terminal" to do script "cd '"$PWD"' && source .venv/bin/activate 2>/dev/null || true && cd src && python train_rl.py"'

sleep 2

echo "launching 2/3: evolutionary strategies..."
osascript -e 'tell app "Terminal" to do script "cd '"$PWD"' && source .venv/bin/activate 2>/dev/null || true && cd src && python evolve.py"'

sleep 2

echo "launching 3/3: hybrid training..."
osascript -e 'tell app "Terminal" to do script "cd '"$PWD"' && source .venv/bin/activate 2>/dev/null || true && cd src && python train_hybrid.py"'

echo ""
echo "✅ all experiments launched!"
echo ""
echo "📊 view progress at: https://wandb.ai"
echo "💾 checkpoints saved to: checkpoints/"
echo ""
echo "to stop: close the terminal windows or ctrl+c in each"
