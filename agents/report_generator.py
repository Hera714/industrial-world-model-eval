"""从既有评测产物确定性生成分析报告；缺失数据明确标记，不推断原因。"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIGURES = (
    ('误差—预测步长', 'reports/week6/results/residual_mlp/figures/open_loop_error_with_residual.png'),
    ('OOD 与观测故障', 'reports/week6/results/degradation/weather_minus_reference.png'),
    ('闭环控制成本', 'reports/week6/results/residual_mlp/figures/closed_loop_kpis_with_residual.png'),
)


def _hash(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def generate(root, output, run_directory=None):
    """读取 Week 6 结果和指定 run，输出可溯源 Markdown；不修改指标。"""
    root, output = Path(root).resolve(), Path(output).resolve()
    lines = ['# 自动失败分析报告', '', '本报告根据已有产物生成，不推断相关性的因果关系。',
             '实验口径与统计单位见 [Week 6 报告](../week6/week6.md)。', '']
    for title, relative in FIGURES:
        path = root / relative
        lines += [f'## {title}', '']
        if path.is_file():
            link = Path(os.path.relpath(path, output.parent)).as_posix()
            lines += [f'![{title}]({link})', '', f'来源：`{relative}`；SHA256：`{_hash(path)}`。', '']
        else:
            lines += [f'缺失：`{relative}`；不能分析该项。', '']
    source = root / 'reports/week6/results/summary/control_by_model.csv'
    lines += ['## 闭环成本与 fallback 示例', '']
    if source.is_file():
        with source.open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        required = ('model', 'domain', 'condition', 'cost_mean', 'cost_count', 'fallback_count_mean')
        if rows and all(key in rows[0] for key in required):
            lines += ['| 模型 | 场景 | 条件 | 平均成本 | 回合数 | 平均 fallback 次数 |',
                      '| --- | --- | --- | ---: | ---: | ---: |']
            for row in rows[:12]:
                lines.append('| ' + ' | '.join(str(row[k]).replace('|', r'\|') for k in required) + ' |')
            lines += ['', '仅展示前 12 行，不能作为完整排名。',
                      f'来源：`{source.relative_to(root).as_posix()}`；SHA256：`{_hash(source)}`。', '']
        else:
            lines += ['数据为空或缺少必要字段，无法生成成本表。', '']
    else:
        lines += ['成本汇总文件缺失，无法生成成本表。', '']
    source = root / 'reports/week6/results/summary/prediction_control_counterexamples.csv'
    lines += ['## 预测—控制不一致案例', '']
    if source.is_file():
        with source.open(newline='', encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        fields = ('episode_id', 'better_prediction_model', 'worse_prediction_model', 'rmse_a', 'rmse_b', 'cost_a', 'cost_b')
        if rows and all(key in rows[0] for key in fields):
            lines += ['| episode | 较低误差模型 | 对照模型 | RMSE A °C | RMSE B °C | 成本 A | 成本 B |',
                      '| --- | --- | --- | ---: | ---: | ---: | ---: |']
            for row in rows[:2]:
                lines.append('| ' + ' | '.join(str(row[key]).replace('|', r'\|') for key in fields) + ' |')
            lines += ['', f'来源：`{source.relative_to(root).as_posix()}`；SHA256：`{_hash(source)}`。', '']
        else:
            lines += ['案例文件为空或字段缺失。', '']
    else:
        lines += ['案例文件缺失，不能推断是否存在不一致。', '']
    lines += ['## 指定运行的失败记录', '']
    if run_directory is None:
        lines += ['未指定 run；没有检查单次运行的失败上下文。', '']
    else:
        run = Path(run_directory).resolve()
        for name in ('status.json', 'failure.json', 'events.jsonl'):
            path = run / name
            if not path.is_file():
                lines.append(f'- `{name}`：不存在；不能据此推断没有失败。')
                continue
            link = Path(os.path.relpath(path, output.parent)).as_posix()
            lines.append(f'- [{name}]({link})：SHA256 `{_hash(path)}`。')
            if name == 'status.json':
                try:
                    status = json.loads(path.read_text(encoding='utf-8')).get('status', '未知')
                    lines.append(f'  状态：`{status}`。')
                except (ValueError, OSError):
                    lines.append('  状态文件无法解析。')
            elif name == 'failure.json':
                try:
                    failure = json.loads(path.read_text(encoding='utf-8'))
                    error_type = str(failure.get('error_type', '未记录')).replace('`', "'")
                    message = str(failure.get('message', '未记录')).replace('`', "'").replace('\n', ' ')
                    context = failure.get('context', {})
                    lines.append(f'  记录的异常类型：`{error_type}`；消息：`{message}`。')
                    lines.append(f'  记录的执行阶段：`{str(context.get("phase", "未记录")).replace(chr(96), chr(39))}`。')
                except (ValueError, OSError, AttributeError):
                    lines.append('  失败文件无法解析；请查看原始文件。')
        lines.append('')
    lines += ['模型偏差、天气或 fallback 的原因需要结合逐步轨迹检查；报告不自动赋予因果结论。', '']
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines), encoding='utf-8')
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='reports/week8/automatic_generated.md')
    parser.add_argument('--run-directory')
    args = parser.parse_args()
    print(generate(ROOT, ROOT / args.output, ROOT / args.run_directory if args.run_directory else None))


if __name__ == '__main__':
    main()
