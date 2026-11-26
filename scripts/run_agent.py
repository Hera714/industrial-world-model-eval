"""总说明：自然语言 Agent 命令入口，模型地址和名称必须明确指定。"""
from pathlib import Path
import argparse,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agents.pipeline import run_agent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--request',required=True)
    parser.add_argument('--endpoint',required=True)
    parser.add_argument('--model',required=True)
    args=parser.parse_args()
    folder,state=run_agent(args.request,endpoint=args.endpoint,model=args.model)
    print(f"Agent 会话：{folder}；状态：{state['status']}")
    if state['status']!='completed':print(state['error'],file=sys.stderr)
    return 0 if state['status']=='completed' else 1

if __name__=='__main__':raise SystemExit(main())
