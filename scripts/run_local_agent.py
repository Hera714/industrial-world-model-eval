"""总说明：校验已下载本地模型文件，执行真实 LLM→BOPTEST→审核演示。"""
from pathlib import Path
import argparse,hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agents.local_llm import LocalGenerator
from agents.pipeline import run_agent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-dir',required=True,help='包含完整模型配置、分词器及权重的目录')
    parser.add_argument('--expected-sha256',required=True,help='上游公布的权重 SHA256')
    parser.add_argument('--revision',required=True,help='上游完整提交 ID')
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda',help='本地推理设备，默认 cuda')
    parser.add_argument('--request',default='运行 BOPTEST，比较 persistence 和 arx')
    args=parser.parse_args();directory=Path(args.model_dir).resolve()
    manifest=json.loads((ROOT/'configs/agent/local_model_manifest.json').read_text())
    if args.revision!=manifest['revision'] or args.expected_sha256!=manifest['weight_sha256']:
        raise ValueError('模型版本或预期哈希与冻结清单不符')
    for name,expected in manifest['metadata_sha256'].items():
        file=directory/name
        if hashlib.sha256(file.read_bytes()).hexdigest()!=expected:
            raise ValueError('模型元数据哈希不匹配: '+name)
    weight=directory/'model.safetensors' 
    digest=hashlib.file_digest(weight.open('rb'),'sha256').hexdigest()
    if digest!=args.expected_sha256:raise ValueError('模型权重 SHA256 不匹配')
    generator=LocalGenerator(str(directory),revision=args.revision,device=args.device)
    folder,state=run_agent(args.request,endpoint='local-transformers',model=str(directory),generator=generator)
    print(json.dumps(dict(session=str(folder),status=state['status'],model_sha256=digest),ensure_ascii=False))
    return int(state['status']!='completed')

if __name__=='__main__':raise SystemExit(main())
