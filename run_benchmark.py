"""总说明：统一 benchmark 命令行入口；配置预检失败不连接仿真，运行异常保留独立 run 失败上下文。"""
from pathlib import Path
import argparse
import sys
from benchmark.runner import run_benchmark


def main():
    parser = argparse.ArgumentParser(description='BOPTEST unified benchmark')
    parser.add_argument('--config', required=True, help='项目内 YAML 配置路径')
    parser.add_argument('--resume-from', help='从已有 run 复用已完成 episode，产生新的 run')
    parser.add_argument('--tracking-config', help='可选 MLflow 本地追踪 YAML')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    try:
        output = run_benchmark(root/args.config, root, resume_from=args.resume_from, tracking_config=(root/args.tracking_config if args.tracking_config else None))
    except Exception as error:
        print(f'Benchmark failed: {type(error).__name__}: {error}', file=sys.stderr)
        return 1
    print('Benchmark completed: ' + str(output), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
