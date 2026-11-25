"""按时间划分数据，并检查滑动窗口的预测目标是否跨界。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def split_by_time(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """排序后按 70%/15%/15% 划分；切分边界只由行数决定。"""
    if "date" not in df:
        raise ValueError("缺少 date 列")
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise")
    if data["date"].isna().any() or data["date"].duplicated().any():
        raise ValueError("时间戳不能缺失或重复")
    data = data.sort_values("date", kind="stable").reset_index(drop=True)
    n = len(data)
    a, b = int(n * 0.70), int(n * 0.85)
    if a == 0 or a == b or b == n:
        raise ValueError("数据不足以分成三个非空集合")
    return {"train": data.iloc[:a].copy(), "val": data.iloc[a:b].copy(), "test": data.iloc[b:].copy()}


def target_windows(splits: dict[str, pd.DataFrame], history: int, horizon: int) -> dict[str, list[pd.DatetimeIndex]]:
    """生成目标时间窗口。验证/测试可使用切分边界之前的历史观测。"""
    if history < 1 or horizon < 1:
        raise ValueError("history 和 horizon 必须为正整数")
    dates = pd.DatetimeIndex(pd.concat([splits[k]["date"] for k in ("train", "val", "test")], ignore_index=True))
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("切分必须按 train、val、test 时间顺序排列，且时间戳唯一")
    result = {}
    offset = 0
    for name in ("train", "val", "test"):
        end = offset + len(splits[name])
        # start 是目标第一个时间点；必须有足够历史，且完整目标留在本集合内。
        starts = range(max(offset, history), end - horizon + 1)
        result[name] = [dates[start:start + horizon] for start in starts]
        if not result[name]:
            raise ValueError(f"{name} 没有完整窗口，请调整历史长度或预测长度")
        offset = end
    validate_target_windows(splits, result)
    return result


def validate_target_windows(splits: dict[str, pd.DataFrame], windows: dict[str, list[pd.DatetimeIndex]]) -> None:
    """验证目标属于各自切分，并检查不同集合的目标时间不重叠。"""
    targets = {}
    for name in ("train", "val", "test"):
        allowed = set(pd.to_datetime(splits[name]["date"]))
        targets[name] = {t for window in windows[name] for t in window}
        if not targets[name] or not targets[name].issubset(allowed):
            raise ValueError(f"{name} 的目标为空或越过切分边界")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if targets[left] & targets[right]:
            raise ValueError(f"{left} 和 {right} 的目标时间重叠")


def split_by_episode(frame, seed=2026):
    """按策略分层打乱完整回合，70/15/15 分配，返回数据及 ID 清单。"""
    if frame.duplicated(['episode_id', 'source_step']).any():
        raise ValueError('Duplicate transition')
    counts = frame.groupby('episode_id').policy.nunique()
    if (counts != 1).any():
        raise ValueError('Episode policy must be unique')
    rng = np.random.default_rng(seed)
    manifest = {name: [] for name in ('train', 'val', 'test')}
    episodes = frame[['episode_id', 'policy']].drop_duplicates()
    for _, group in episodes.groupby('policy', sort=True):
        ids = rng.permutation(sorted(group.episode_id))
        first, second = int(len(ids)*.7), int(len(ids)*.85)
        if first == 0 or first == second or second == len(ids):
            raise ValueError('Insufficient episodes per policy')
        for name, values in zip(manifest, (ids[:first], ids[first:second], ids[second:])):
            manifest[name].extend(values.tolist())
    parts = {name: frame[frame.episode_id.isin(ids)].copy() for name, ids in manifest.items()}
    if sum(map(len, parts.values())) != len(frame):
        raise AssertionError('Split lost transitions')
    for left, right in (('train','val'),('train','test'),('val','test')):
        if set(manifest[left]) & set(manifest[right]):
            raise AssertionError('Episode overlap')
        if set(parts[left].scenario_id) & set(parts[right].scenario_id):
            raise ValueError('Scenario overlap; use grouped scenario assignment')
    return parts, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="按时间切分数据并验证目标窗口")
    parser.add_argument("--data", default="data/raw/energydata_complete.csv")
    parser.add_argument("--history", type=int, default=144)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--output", default="reports/week1_split.json")
    args = parser.parse_args()
    splits = split_by_time(pd.read_csv(args.data))
    windows = target_windows(splits, args.history, args.horizon)
    summary = {"history": args.history, "horizon": args.horizon, "target_overlap_check": "passed", "splits": {}}
    for name, part in splits.items():
        summary["splits"][name] = {"start": part.date.iloc[0].isoformat(), "end": part.date.iloc[-1].isoformat(), "rows": len(part), "windows": len(windows[name]), "first_target": windows[name][0][0].isoformat(), "last_target": windows[name][-1][-1].isoformat()}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"切分记录已保存：{output}")


if __name__ == "__main__":
    main()
