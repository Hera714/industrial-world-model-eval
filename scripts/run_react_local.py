"""总说明：真实本地 LLM 自主选择工具；运行仍受同配置验证和请求白名单约束。"""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agents.local_llm import LocalGenerator
from agents.react_llm import LocalToolDecider,PROMPT
from agents.tools import invoke
from agents.react import run_react

def main():
    manifest=json.loads((ROOT/'configs/agent/local_model_manifest.json').read_text());folder=ROOT/'external/qwen-agent'
    with (folder/manifest['weight']).open('rb') as stream:assert hashlib.file_digest(stream,'sha256').hexdigest()==manifest['weight_sha256']
    for name,digest in manifest['metadata_sha256'].items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
    model=LocalGenerator(str(folder),manifest['revision']);decide=LocalToolDecider(model)
    result=run_react('在 BOPTEST 中测试 arx',decide,max_turns=8,tool=invoke)
    output=ROOT/'reports/week8/react_real_execution.json';output.write_text(json.dumps(dict(model=manifest['repository'],revision=manifest['revision'],
        model_sha256=manifest['weight_sha256'],prompt=PROMPT,prompt_sha256=hashlib.sha256(PROMPT.encode()).hexdigest(),
        temperature=0,mode='real_boptest',result=result),ensure_ascii=False,indent=2))
    print(output)
if __name__=='__main__':main()
