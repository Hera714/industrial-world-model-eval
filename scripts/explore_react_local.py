"""总说明：使用真实本地模型做只读工具选择探索，禁止在无审核任务集时启动仿真。"""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agents.local_llm import LocalGenerator
from agents.react_llm import LocalToolDecider
from agents.tools import invoke
from agents.react import run_react

def safe_tool(name,args):
    if name=='run_benchmark':return dict(ok=False,error=dict(code='DRY_RUN',message='该探索禁止执行仿真'),result=None)
    return invoke(name,args)

def main():
    manifest=json.loads((ROOT/'configs/agent/local_model_manifest.json').read_text());folder=ROOT/'external/qwen-agent'
    with (folder/manifest['weight']).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==manifest['weight_sha256']
    for name,digest in manifest['metadata_sha256'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
    model=LocalGenerator(str(folder),manifest['revision']);decide=LocalToolDecider(model)
    result=run_react('在 BOPTEST 中测试 arx',decide,max_turns=4,tool=safe_tool)
    output=ROOT/'reports/week8/react_real_exploration.json';output.write_text(json.dumps(dict(model=manifest['repository'],revision=manifest['revision'],mode='dry_run',result=result),ensure_ascii=False,indent=2))
    print(output)
if __name__=='__main__':main()
