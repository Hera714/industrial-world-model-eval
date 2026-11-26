"""总说明：验证报告在数据缺失时明确报告，不捏造指标。"""
import tempfile,unittest
from evaluators.week8_report import generate_report
class ReportTest(unittest.TestCase):
    def test_missing_inputs(self):
        with tempfile.TemporaryDirectory() as path:
            report=generate_report(path)
            self.assertEqual(report.count('数据缺失'),5)
            self.assertIn('不能推断没有失败',report)
if __name__=='__main__':unittest.main()
