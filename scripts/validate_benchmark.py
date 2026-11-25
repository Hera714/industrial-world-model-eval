"""总说明：统一配置离线验收入口，成功后才允许下一任务Runner启动环境。"""
from pathlib import Path
import sys,argparse,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from benchmark.config import load_config,ConfigurationError


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    try:
        config=load_config(ROOT/args.config,ROOT)
    except (ConfigurationError,OSError,ValueError) as error:
        print(json.dumps(dict(status='failed',error=str(error)),ensure_ascii=False),file=sys.stderr)
        return 2
    print(json.dumps(dict(status='passed',config=args.config,models=len(config['model']['instances']),
        episodes=len(config['dataset']['evaluation_episodes']),network_used=False),ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
