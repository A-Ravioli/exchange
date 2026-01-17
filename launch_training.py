#!/usr/bin/env python3
"""launch all three training experiments"""

import subprocess
import os
import sys

print("🚀 preparing to launch all training experiments...")
print("=" * 60)

# setup
base_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(base_dir, "src")

# check if wandb is configured
try:
    import wandb
    print("✅ wandb found")
    
    # check if logged in
    api = wandb.Api()
    print(f"✅ wandb logged in")
except:
    print("⚠️  wandb not configured. run: wandb login")
    sys.exit(1)

# create checkpoint directories
os.makedirs("checkpoints/rl_selfplay", exist_ok=True)
os.makedirs("checkpoints/evolution", exist_ok=True)
os.makedirs("checkpoints/hybrid", exist_ok=True)
print("✅ checkpoint directories created")

print("=" * 60)
print("")
print("launching three training experiments:")
print("  1. RL Self-Play (100k iterations)")
print("  2. Evolution (50k generations)")  
print("  3. Hybrid (50k iterations)")
print("")
print("this will run for DAYS. check wandb for progress.")
print("press ctrl+c in this window to stop all experiments.")
print("")
print("=" * 60)

# use screen or tmux if available, otherwise use simple background processes
try:
    # launch all three in the background with nohup
    scripts = [
        ("rl_selfplay", "train_rl.py"),
        ("evolution", "evolve.py"),
        ("hybrid", "train_hybrid.py")
    ]
    
    processes = []
    
    for name, script in scripts:
        log_file = f"logs/{name}.log"
        os.makedirs("logs", exist_ok=True)
        
        cmd = f"cd {src_dir} && python {script} > ../{log_file} 2>&1"
        proc = subprocess.Popen(cmd, shell=True)
        processes.append((name, proc))
        print(f"✅ launched {name} (pid: {proc.pid}, log: {log_file})")
    
    print("")
    print("=" * 60)
    print("all experiments running!")
    print("📊 view progress: https://wandb.ai")
    print("📁 logs: logs/")
    print("💾 checkpoints: checkpoints/")
    print("")
    print("to stop all experiments:")
    print(f"  kill {' '.join(str(p.pid) for _, p in processes)}")
    print("=" * 60)
    
except KeyboardInterrupt:
    print("\n⚠️  interrupted - stopping all experiments...")
    for name, proc in processes:
        proc.terminate()
    print("✅ all experiments stopped")
