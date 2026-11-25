"""总说明：记录单元压力测试，并在真实BOPTEST上连续验证三类规划故障的释放动作。"""
from pathlib import Path
import sys,json,unittest
import yaml
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tests.test_week6_control import Week6FallbackTest,plan_fault
from simulators.boptest_control import BoptestControlEnv


def main():
    output=ROOT/'reports/week6/results/fallback_stress';output.mkdir(exist_ok=True)
    with (output/'unit_tests.log').open('w') as log:
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Week6FallbackTest))
    assert result.wasSuccessful()
    cfg=yaml.safe_load((ROOT/'configs/experiments/week6_boptest.yaml').read_text());cfg.update(start_time=40*86400,episode_steps=6)
    env=BoptestControlEnv(cfg);rows=[]
    try:
        env.reset(seed=42)
        for fault,reason in [('timeout','timeout'),('negative_energy','no_valid_candidates'),('exception','planning_error')]:
            override=env.step([23.])[-1]
            assert override['override_active']
            action,detail=plan_fault(fault)
            assert detail['fallback_reason']==reason
            obs,reward,terminated,truncated,info=env.safe_step(action,reason)
            assert not info['override_active'] and info['requested_action_c'] is None
            assert info['fallback_reason']==reason
            assert env.steps==len(rows)*2+2
            activation=env.request('PUT',f'/results/{env.testid}',dict(
                point_names=['oveTSet_activate','oveTSet_u','reaTSetHea_y'],
                start_time=info['timestamp'],final_time=info['timestamp']))
            assert not bool(activation['oveTSet_activate'][-1])
            rows.append(dict(fault=fault,planning=detail,prior_override=override,executed_fallback=info,
                server_confirmation=activation,
                action_semantics='Release override; embedded controller selects actual setpoint. No fixed fallback temperature assumed.'))
        assert truncated and not terminated
        (output/'real_api_verification.json').write_text(json.dumps(dict(status='passed',unit_tests=result.testsRun,
            real_fault_cases=3,real_steps=6,rows=rows,scope='Planning failures; HTTP advance failure is not retried as a fallback'),indent=2))
        print('4 unit stress tests and 3 real API fallback cases passed',flush=True)
    finally:
        env.close()


if __name__=='__main__':main()
