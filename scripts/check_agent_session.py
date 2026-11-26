"""总说明：审核指定 Agent 会话并记录各项结果，不修改原始会话。"""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agents.session_auditor import audit_session

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--session',required=True);parser.add_argument('--output')
    args=parser.parse_args();session=(ROOT/args.session).resolve();base=(ROOT/'reports/week8/sessions').resolve()
    if not session.is_relative_to(base):parser.error('session 必须位于 reports/week8/sessions')
    result=audit_session(session);payload=json.dumps(result,ensure_ascii=False,indent=2)
    if args.output:
        output=(ROOT/args.output).resolve()
        if not output.is_relative_to(ROOT):parser.error('output 必须位于项目目录')
        output.parent.mkdir(parents=True,exist_ok=True);output.write_text(payload)
    print(payload)
    return int(not result['passed'])

if __name__=='__main__':raise SystemExit(main())
