"""Week 1 三个直接多步预测基线：先验证选择，再评估测试集。"""
import json
from pathlib import Path
import numpy as np
import torch
from metrics.prediction import mae, rmse, error_by_horizon
from models.persistence import Persistence
from models.arx import LinearAutoregression
from models.learned_dynamics import MLPForecast

DATA = Path("data/processed/week1")
OUT = Path("reports/week1_baselines")

def load(name):
    with np.load(DATA / f"{name}.npz", allow_pickle=False) as f:
        return {k: f[k] for k in f.files}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((DATA / "metadata.json").read_text(encoding="utf-8"))
    train, val = load("train"), load("val")
    horizon = meta["horizon"]; target = meta["target"]; target_index = meta["features"].index(target)
    models = {"persistence": Persistence(horizon, target_index), "linear": LinearAutoregression()}
    models.update({f"mlp_{seed}": MLPForecast(horizon=horizon, seed=seed) for seed in (42, 43, 44)})
    for name, model in models.items():
        print(f"训练/验证 {name}", flush=True)
        model.fit(train["X"], train["y"], val["X"], val["y"])
    # 所有模型和参数确定后才加载测试集。
    test = load("test"); predictions = {}
    mean = meta["scaler"]["mean"][target]; scale = meta["scaler"]["scale"][target]
    actual = test["y"] * scale + mean
    scores = {}
    for name, model in models.items():
        p = model.predict(test["X"])
        if p.shape != test["y"].shape or not np.isfinite(p).all():
            raise ValueError(f"{name} 预测形状或数值不合法: {p.shape}")
        predictions[name] = p * scale + mean
        h_mae, h_rmse = error_by_horizon(actual, predictions[name])
        scores[name] = {"mae_c": mae(actual, predictions[name]), "rmse_c": rmse(actual, predictions[name]),
                        "horizons": {str(h): {"mae_c": float(h_mae[h-1]), "rmse_c": float(h_rmse[h-1])} for h in (1, 6, 12, 24) if h <= horizon}}
        if name.startswith("mlp_"): torch.save(model.model.state_dict(), OUT / f"{name}.pt")
    np.savez_compressed(OUT / "test_predictions.npz", actual=actual, target_start=test["target_start"], **predictions)
    config = {"prediction_mode": "direct multi-step", "history": meta["history"], "horizon": horizon,
              "features": meta["features"], "target": target, "test_windows": len(actual),
              "linear": {"candidate_alphas": list(models["linear"].alphas), "selected_alpha": models["linear"].alpha, "validation_mse": models["linear"].validation_mse},
              "mlp": {name: {"seed": model.seed, "hidden": model.hidden, "lr": model.lr, "max_epochs": model.epochs, "patience": model.patience, "batch_size": model.batch_size, "best_epoch": model.best_epoch, "validation_mse": model.validation_mse, "device": model.device} for name, model in models.items() if name.startswith("mlp_")},
              "torch_version": torch.__version__}
    (OUT / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (OUT / "metrics.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps(scores, indent=2))

if __name__ == "__main__": main()
