"""总说明：注入 episode 故障验证续跑、哈希拒绝及原有失败记录保留。"""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from benchmark.config import load_config
from benchmark.runner import run_benchmark
from evaluators.benchmark_boptest import run_session
from tests.test_benchmark_runner import MemoryEnv

ROOT=Path(__file__).resolve().parents[1]

class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT)
        self.addCleanup(self.temp.cleanup)
        self.cfg=load_config(ROOT/'configs/benchmark/boptest_smoke.yaml',ROOT)
        self.cfg['output']['directory']=str(Path(self.temp.name).relative_to(ROOT))
        MemoryEnv.instances=[];MemoryEnv.fail_step=None

    def interrupted(self):
        def fail_arx(config, episode, recorded, model, name, condition, emit, progress, env_factory):
            if name=='arx': raise RuntimeError('injected failure')
            return run_session(config,episode,recorded,model,name,condition,emit,progress,env_factory)
        with patch('benchmark.runner.run_session',side_effect=fail_arx):
            with self.assertRaisesRegex(RuntimeError,'injected failure'):
                run_benchmark(self.cfg,ROOT,MemoryEnv)
        return next(Path(self.temp.name).iterdir())

    def test_resume_reuses_only_complete_episodes(self):
        source=self.interrupted()
        before=json.loads((source/'failure.json').read_text())
        MemoryEnv.instances=[]
        result=run_benchmark(self.cfg,ROOT,MemoryEnv,resume_from=source)
        self.assertNotEqual(source,result)
        self.assertEqual(len(MemoryEnv.instances),2) # ARX 与 embedded，采集和 persistence 已完成。
        self.assertEqual(json.loads((source/'failure.json').read_text()),before)
        self.assertEqual(json.loads((result/'status.json').read_text())['status'],'completed')
        events=(result/'events.jsonl').read_text()
        self.assertEqual(events.count('episode_reused'),2)
        metrics=json.loads((result/'metrics.json').read_text())['metrics']
        self.assertEqual(len(metrics['closed_loop']),3)

    def test_configuration_mismatch_rejected_before_environment(self):
        source=self.interrupted();changed=copy.deepcopy(self.cfg)
        changed['horizon']['candidates']=33
        MemoryEnv.instances=[]
        with self.assertRaisesRegex(ValueError,'configuration mismatch'):
            run_benchmark(changed,ROOT,MemoryEnv,resume_from=source)
        self.assertEqual(MemoryEnv.instances,[])

    def test_corrupt_completed_episode_rejected(self):
        source=self.interrupted()
        payload=next((source/'episodes').glob('*/episode.json'))
        payload.write_text(payload.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'integrity mismatch'):
            run_benchmark(self.cfg,ROOT,MemoryEnv,resume_from=source)

    def test_input_hash_mismatch_rejected(self):
        source=self.interrupted()
        path=source/'input_hashes.json';data=json.loads(path.read_text())
        key=next(iter(data['files']));data['files'][key]='0'*64;path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError,'input hash mismatch'):
            run_benchmark(self.cfg,ROOT,MemoryEnv,resume_from=source)

if __name__=='__main__': unittest.main()
