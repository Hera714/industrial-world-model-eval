"""总说明：本地真实模型与 SQLite LangGraph 命令入口，审批需独立命令和配置摘要。"""
from pathlib import Path
import argparse,hashlib,json,sqlite3,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from agents.graph import build_graph
from agents.local_llm import LocalGenerator

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--thread',required=True)
    parser.add_argument('--request');parser.add_argument('--approve-hash');parser.add_argument('--reject',action='store_true')
    parser.add_argument('--db',default='reports/week8/graph.sqlite');parser.add_argument('--model-dir',default='external/qwen-agent')
    args=parser.parse_args()
    if sum([bool(args.request),bool(args.approve_hash),args.reject])!=1:parser.error('必须选择 request、approve-hash 或 reject 之一')
    db=(ROOT/args.db).resolve()
    if not db.is_relative_to(ROOT):parser.error('数据库必须位于项目目录')
    db.parent.mkdir(parents=True,exist_ok=True)
    model_path=(ROOT/args.model_dir).resolve();manifest=json.loads((ROOT/'configs/agent/local_model_manifest.json').read_text())
    # 恢复审批不需要重新加载 LLM；仅生成节点实际调用 generator。
    def generate(request,**kwargs):
        for name,expected in manifest['metadata_sha256'].items():
            if hashlib.sha256((model_path/name).read_bytes()).hexdigest()!=expected:raise ValueError('模型元数据哈希不匹配')
        with (model_path/manifest['weight']).open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=manifest['weight_sha256']:raise ValueError('模型权重哈希不匹配')
        return LocalGenerator(str(model_path),manifest['revision'])(request,**kwargs)
    config={'configurable':{'thread_id':args.thread}}
    with sqlite3.connect(db,check_same_thread=False) as connection:
        graph=build_graph(generator=generate,checkpointer=SqliteSaver(connection))
        if args.request:
            if graph.get_state(config).values:raise ValueError('线程已存在；请选择新 thread')
            value=dict(request=args.request,endpoint='local',model=str(model_path),require_approval=True)
        else:value=Command(resume={'approved':not args.reject,'config_sha256':args.approve_hash})
        result=graph.invoke(value,config)
        if '__interrupt__' in result:
            print(json.dumps(dict(status='awaiting_approval',details=result['__interrupt__'][0].value),ensure_ascii=False,indent=2))
        else:print(json.dumps(result,ensure_ascii=False,indent=2,default=str))
        return int(result.get('status') in ('failed','rejected'))

if __name__=='__main__':raise SystemExit(main())
