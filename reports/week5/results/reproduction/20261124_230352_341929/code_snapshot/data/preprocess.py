"""Week 1：时间对齐、无未来泄漏的预处理及直接多步预测窗口。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from .split import split_by_time, target_windows
except ImportError:
    from split import split_by_time, target_windows

FEATURES = ["T1", "T2", "T3", "RH_1", "T_out"]
TARGET = "T1"
NAMES = ("train", "val", "test")


def prepare_frame(df, features=FEATURES, frequency="10min", fill_limit=6):
    """检查时间网格并标记缺失；仅以前向填充使用过去观测。"""
    if fill_limit < 1:
        raise ValueError("fill_limit 必须为正数")
    required = ["date", *features]
    missing_cols = set(required) - set(df.columns)
    if missing_cols:
        raise ValueError(f"缺少字段: {sorted(missing_cols)}")
    frame = df[required].copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    if frame.date.isna().any() or frame.date.duplicated().any() or frame.empty:
        raise ValueError("时间戳不能为空、重复，数据不能为空")
    frame = frame.sort_values("date").set_index("date")
    grid = pd.date_range(frame.index[0], frame.index[-1], freq=frequency)
    if not frame.index.isin(grid).all():
        raise ValueError("存在不在固定时间网格上的记录；请先检查采样时间")
    for col in features:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if np.isinf(frame[col].to_numpy()).any():
            raise ValueError(f"{col} 含无穷值")
    missing_timestamps = len(grid) - len(frame)
    aligned = frame.reindex(grid)
    aligned.index.name = "date"
    flags = aligned[features].isna().astype(np.uint8)
    flags.columns = [f"{c}_missing" for c in features]
    # 真实目标不填补；缺失目标的窗口将被丢弃。
    target_observed = aligned[TARGET].notna().to_numpy()
    filled = aligned[features].ffill(limit=fill_limit)
    return filled.reset_index(), flags.reset_index(drop=True), target_observed, {
        "frequency": frequency, "inserted_timestamps": missing_timestamps,
        "missing_before_fill": {c: int(aligned[c].isna().sum()) for c in features},
        "missing_after_fill": {c: int(filled[c].isna().sum()) for c in features},
        "forward_fill_limit_steps": fill_limit,
    }


def build_dataset(df, history=144, horizon=24, features=FEATURES, fill_limit=6):
    """仅在训练时段拟合标准化，返回各集合窗口、缺失标记和元数据。"""
    if TARGET not in features:
        raise ValueError("特征必须包含目标 T1")
    frame, flags, observed, diagnostics = prepare_frame(df, features, fill_limit=fill_limit)
    splits = split_by_time(frame)
    # 重用切分模块检查所有候选目标窗口均属于对应区间。
    candidates = target_windows(splits, history, horizon)
    train_n = len(splits["train"])
    train = frame.loc[:train_n - 1, features]
    means = train.mean()
    scales = train.std(ddof=0).replace(0, 1.0)
    if means.isna().any() or scales.isna().any():
        raise ValueError("训练集存在无法计算标准化参数的特征")
    values = ((frame[features] - means) / scales).to_numpy(dtype=np.float32)
    missing_flags = flags.to_numpy(dtype=np.uint8)
    times = pd.DatetimeIndex(frame.date)
    target_idx = features.index(TARGET)
    result = {}
    summary = {"history": history, "horizon": horizon, "features": features, "target": TARGET,
               "diagnostics": diagnostics, "scaler": {"fit_partition": "train", "mean": means.to_dict(), "scale": scales.to_dict()}, "splits": {}}
    for name in NAMES:
        X, Y, masks, starts = [], [], [], []
        rejected = 0
        for target_times in candidates[name]:
            start = times.get_loc(target_times[0])
            x = values[start-history:start]
            y = values[start:start+horizon, target_idx]
            # 目标必须是真实观测，输入必须在有限前向填充后可用。
            if not np.isfinite(x).all() or not observed[start:start+horizon].all():
                rejected += 1
                continue
            X.append(x); Y.append(y); masks.append(missing_flags[start-history:start]); starts.append(start)
        if not X:
            raise ValueError(f"{name} 没有可用窗口")
        result[name] = {"X": np.stack(X), "y": np.stack(Y), "missing_mask": np.stack(masks),
                        "target_start": times[np.array(starts)].to_numpy(dtype="datetime64[ns]")}
        part = splits[name]
        summary["splits"][name] = {"start": part.date.iloc[0].isoformat(), "end": part.date.iloc[-1].isoformat(),
            "rows": len(part), "candidate_windows": len(candidates[name]), "windows": len(X), "discarded_windows": rejected}
    return result, summary


def main():
    parser = argparse.ArgumentParser(description="构造无未来泄漏的时序预测窗口")
    parser.add_argument("--data", default="data/raw/energydata_complete.csv")
    parser.add_argument("--output", default="data/processed/week1")
    parser.add_argument("--history", type=int, default=144)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()
    datasets, summary = build_dataset(pd.read_csv(args.data), args.history, args.horizon)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    for name, arrays in datasets.items():
        np.savez_compressed(out / f"{name}.npz", **arrays)
    (out / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"已保存至 {out}")


if __name__ == "__main__":
    main()
