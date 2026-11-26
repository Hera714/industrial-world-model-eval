"""总说明：真实 LLM 的受控校验失败实验；故障注入明确标记，不运行模拟器。"""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agents.local_llm import LocalGenerator
from agents.react_llm import LocalToolDecider
from agents.react import run_react
from agents.tools import invoke

def main():
    manifest=json.loads((ROOT/'configs/agent/local_model_manifest.json').read_text());folder=ROOT/'external/qwen-agent'
    with (folder/manifest['weight']).open('rb') as stream:
        if hashlib.file_digest(stream,'sha256').hexdigest()!=manifest['weight_sha256']:raise ValueError('模型权重不匹配')
    for name,digest in manifest['metadata_sha256'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=digest:raise ValueError('元数据不匹配')
    generator=LocalGenerator(str(folder),manifest['revision']);decide=LocalToolDecider(generator)
    attempts=0
    def tool(name,args):
        nonlocal attempts
        if name=='validate_config':
            attempts+=1
            if attempts==1:return dict(ok=False,result=None,error=dict(code='INJECTED_INVALID',message='测试注入：请重新检查模型 arx 并重新调用 validate_config'))
        if name=='run_benchmark':return dict(ok=False,result=None,error=dict(code='DRY_RUN',message='仅测试，不执行仿真'))
        return invoke(name,args)
    result=run_react('在 BOPTEST 中测试 arx',decide,max_turns=5,tool=tool)
    events=result['history']
    corrected=any(e['tool']=='validate_config' and e['observation'].get('ok') for e in events[1:])
    output=dict(model=manifest['repository'],revision=manifest['revision'],test='首次校验故障注入，真实模型自主选择后续动作',
                simulated_fault=True,real_simulator=False,corrected=corrected,result=result)
    path=ROOT/'reports/week8/react_validation_correction.json';path.write_text(json.dumps(output,ensure_ascii=False,indent=2))
    print(json.dumps(dict(corrected=corrected,status=result['status'],path=str(path)),ensure_ascii=False))

if __name__=='__main__':main()
