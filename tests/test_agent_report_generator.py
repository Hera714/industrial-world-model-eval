"""检查报告明确表示缺失产物，并引用与哈希真实产物。"""
import tempfile
import unittest
from pathlib import Path
from agents.report_generator import generate


class ReportGeneratorTest(unittest.TestCase):
    def test_missing_inputs_are_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = generate(root, root / 'out/report.md')
            text = output.read_text(encoding='utf-8')
            self.assertIn('缺失', text)
            self.assertIn('未指定 run', text)
            self.assertNotIn('![误差—预测步长]', text)

    def test_uses_actual_file_and_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / 'reports/week6/results/degradation/weather_minus_reference.png'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'fixture')
            run = root / 'run'
            run.mkdir()
            (run / 'status.json').write_text('{"status":"failed"}', encoding='utf-8')
            text = generate(root, root / 'reports/week8/report.md', run).read_text(encoding='utf-8')
            self.assertIn('weather_minus_reference.png', text)
            self.assertIn('SHA256', text)
            self.assertIn('状态：`failed`', text)
            self.assertIn('failure.json`：不存在', text)

    def test_failure_message_is_extracted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'run'
            run.mkdir()
            (run / 'failure.json').write_text('{"error_type":"RuntimeError","message":"timed out","context":{"phase":"reset"}}')
            text = generate(root, root / 'report.md', run).read_text(encoding='utf-8')
            self.assertIn('RuntimeError', text)
            self.assertIn('timed out', text)
            self.assertIn('reset', text)


if __name__ == '__main__':
    unittest.main()
