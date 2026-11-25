"""总说明：保持一个真实测试实例存活，校验服务端实际下载 FMU 与冻结本地 FMU 的 SHA256。"""
from pathlib import Path
import sys,json,subprocess,hashlib
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from simulators.boptest_control import BoptestControlEnv


def main():
    cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text());cfg.update(start_time=86400*40,episode_steps=1)
    env=BoptestControlEnv(cfg)
    try:
        env.reset(seed=42);remote=f'/simulate/{env.testid}/model.fmu';found=[]
        names=subprocess.check_output(['docker','ps','--filter','name=project1-boptest-worker','--format','{{.Names}}'],text=True).splitlines()
        expected=hashlib.sha256((ROOT/'external/project1-boptest/testcases/bestest_hydronic_heat_pump/models/wrapped.fmu').read_bytes()).hexdigest()
        for name in names:
            result=subprocess.run(['docker','exec',name,'sha256sum',remote],capture_output=True,text=True)
            if result.returncode==0:
                actual=result.stdout.split()[0];assert actual==expected
                found.append(dict(container=name,testid=env.testid,remote_path=remote,sha256=actual))
        assert len(found)==1,'Could not identify unique live worker FMU'
        (ROOT/cfg['output']/'assets/deployed_fmu_check.json').write_text(json.dumps(dict(status='passed',expected_sha256=expected,instances=found),indent=2))
        print('Deployed FMU equals frozen local FMU',flush=True)
    finally:env.close()


if __name__=='__main__':main()
