"""逐种子核对正则前后网络权重是否相同，保存可审计结果。"""
from pathlib import Path
import json
import torch

root=Path(__file__).resolve().parents[1]
checks=[]
for seed in (42,43,44):
    filename=f'mlp_hidden64_seed{seed}.pt'
    left=torch.load(root/'reports/week4/results/training_residual'/filename,weights_only=True)
    right=torch.load(root/'reports/week4/results/training_rate_1'/filename,weights_only=True)
    checks.append(dict(seed=seed,identical=all(torch.equal(value,right['weights'][key]) for key,value in left['weights'].items())))
path=root/'reports/week4/results/training_rate_1/weight_comparison.json'
path.write_text(json.dumps(checks,indent=2))
print(checks)
