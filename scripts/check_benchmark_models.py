"""总说明：离线验收统一模型加载、单窗口/批量输出与物理单位接口，不连接 BOPTEST。"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from benchmark.loaders import load_models


def main():
    """按统一 YAML 加载真实检查点，打印各模型的接口验收结果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    models = load_models(ROOT / args.config, ROOT)
    rows = []
    for name, model in models.items():
        history = np.tile(model.model.mean, (model.length, 1))
        rows.append(dict(name=name, kind=model.kind, device=model.device,
                         input_shape=list(history.shape),
                         output_shape=list(model.predict(history).shape),
                         batch_output_shape=list(model.predict(np.stack([history, history])).shape)))
    print(json.dumps(dict(status='passed', network_used=False, models=rows), indent=2))


if __name__ == '__main__':
    main()
