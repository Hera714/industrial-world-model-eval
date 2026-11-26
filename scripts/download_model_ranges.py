"""总说明：按固定字节范围续传权重，逐段校验响应长度与位置，最终验证 SHA256。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import argparse,hashlib,os,subprocess,time

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--url',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--size',required=True,type=int);parser.add_argument('--sha256',required=True)
    parser.add_argument('--proxy',default=None);parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args();target=Path(args.output);parts=target.parent/(target.name+'.parts');parts.mkdir(parents=True,exist_ok=True)
    chunk=8*1024*1024;count=(args.size+chunk-1)//chunk
    def get(index):
        start=index*chunk;end=min(args.size,start+chunk)-1;file=parts/f'{index:05d}';length=end-start+1
        if file.exists() and file.stat().st_size==length:return
        partial=file.with_suffix('.partial');header=file.with_suffix('.header')
        for attempt in range(10):
            command=['curl','-fsSL','--connect-timeout','20','--max-time','120','-H',f'Range: bytes={start}-{end}','-D',str(header),'-o',str(partial)]
            if args.proxy:command+=['--proxy',args.proxy,'--noproxy','']
            command.append(args.url)
            result=subprocess.run(command,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
            headers=header.read_text(errors='replace').lower() if header.exists() else ''
            expected=f'content-range: bytes {start}-{end}/{args.size}'
            if result.returncode==0 and expected in headers and partial.exists() and partial.stat().st_size==length:
                os.replace(partial,file);print(f'chunk {index+1}/{count}',flush=True);return
            time.sleep(min(attempt+1,5))
        raise RuntimeError(f'无法验证分段 {index}')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:list(pool.map(get,range(count)))
    # 合并到新文件，避免覆盖尚未完成或用户已有文件。
    merged=target.with_suffix(target.suffix+'.verified.part');digest=hashlib.sha256()
    with merged.open('wb') as out:
        for index in range(count):
            with (parts/f'{index:05d}').open('rb') as source:
                for block in iter(lambda:source.read(1024*1024),b''):
                    digest.update(block);out.write(block)
    if digest.hexdigest()!=args.sha256:raise ValueError('整体 SHA256 不匹配，保留分段供核查')
    os.replace(merged,target);print('verified '+digest.hexdigest(),flush=True)

if __name__=='__main__':main()
