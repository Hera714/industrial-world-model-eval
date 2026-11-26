"""总说明：从 Week6 已保存图表和指标生成可追溯报告，不生成未计算的数据或原因。"""
from pathlib import Path
import csv,hashlib

ROOT=Path(__file__).resolve().parents[1]

def generate_report(root=ROOT):
    root=Path(root);base=root/'reports/week6/results'
    sections=[('误差—预测步长','residual_mlp/figures/open_loop_error_with_residual.png'),
              ('OOD 与观测故障','degradation/weather_minus_reference.png'),
              ('闭环控制成本','residual_mlp/figures/closed_loop_kpis_with_residual.png')]
    lines=['# 自动分析报告','','本报告引用既有实验产物；具体实验条件与统计单位见 Week 6 协议。','']
    for title,relative in sections:
        file=base/relative;lines += ['## '+title,'']
        if file.is_file():lines += [f'![{title}](../week6/results/{relative})',f'SHA256：`{hashlib.sha256(file.read_bytes()).hexdigest()}`']
        else:lines += ['数据缺失：未找到对应图表。']
        lines.append('')
    table=base/'summary/control_by_model.csv'
    lines += ['## 闭环结果示例','']
    if table.exists():
        with table.open(newline='',encoding='utf-8') as stream:rows=list(csv.DictReader(stream))
        lines += ['| 模型 | 场景 | 条件 | 平均成本 | 回合数 |','| --- | --- | --- | ---: | ---: |']
        for row in rows[:12]:lines.append(f"| {row['model']} | {row['domain']} | {row['condition']} | {row['cost_mean']} | {row['cost_count']} |")
        lines += ['',f'来源：`{table.relative_to(root)}`；展示前 12 行，不作为全部结果排名。']
    else:lines += ['数据缺失：控制指标表不存在。']
    cases=base/'summary/prediction_control_counterexamples.csv'
    lines += ['', '## 可追溯预测—控制反例', '']
    if cases.is_file():
        with cases.open(newline='',encoding='utf-8') as stream: examples=list(csv.DictReader(stream))
        if examples:
            lines += ['| 时间块 | 较低 RMSE 模型 | 另一模型 | RMSE °C | 成本 | fallback 次数 |',
                      '| --- | --- | --- | --- | --- | --- |']
            for row in examples[:2]:
                lines.append(f"| {row['episode_id']} | {row['better_prediction_model']} | {row['worse_prediction_model']} | {row['rmse_a']} / {row['rmse_b']} | {row['cost_a']} / {row['cost_b']} | {row['fallback_a']} / {row['fallback_b']} |")
            lines += ['',f'来源：`{cases.relative_to(root)}`。较低开环误差与更高闭环成本同时出现；这是配对观察，不证明单一因果原因。']
        else:lines += ['没有记录反例；不能由此推断所有模型排序一致。']
    else:lines += ['数据缺失：反例表不存在。']
    lines += ['', '## 错误原因与限制','', '本报告只引用已保存证据。模型偏差、扰动或 fallback 是否解释某个案例，应查看对应逐步轨迹；不能仅凭图形推断因果。',
              '实际模型/任务失败事件请查看对应 run 的 `events.jsonl` 和 `failure.json`。若文件不存在，不能推断没有失败。','']
    return '\n'.join(lines)

def main():
    target=ROOT/'reports/week8/automatic_analysis.md';target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(generate_report(),encoding='utf-8');print(target)

if __name__=='__main__':main()
