"""总说明：测试 Agent 数据集审核能发现重复身份和错误答案。"""
import json,tempfile,unittest
from pathlib import Path
from evaluators.agent_dataset_audit import audit_cases
class DatasetAuditTest(unittest.TestCase):
    def test_detects_duplicate_and_unknown_model(self):
        item=dict(id='same',request='test',expected_action='generate_and_validate',expected_models=['unknown'],expected_tools=[],rubric={})
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'cases.json';path.write_text(json.dumps({'cases':[item,item]}))
            result=audit_cases(path,['arx'])
            self.assertEqual(result['status'],'failed')
            self.assertTrue({'User','Environment','Ground Truth','Evaluation'} <= {i['component'] for i in result['issues']})
if __name__=='__main__':unittest.main()
