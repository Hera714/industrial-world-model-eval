"""独立验证楼宇 FMU 的平台、初始化和室温输出。"""
import argparse
import json
from pathlib import Path

from fmpy import read_model_description, simulate_fmu, supported_platforms
from importlib.metadata import version

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("fmu", type=Path)
args = parser.parse_args()
# 此上游 FMU 的部分内部变量未通过 FMPy 严格 XML 校验。
model = read_model_description(str(args.fmu), validate=False)
platforms = supported_platforms(str(args.fmu))
points = [var for var in model.modelVariables if var.causality == "output" and "tzon" in var.name.lower()]
print(json.dumps(dict(fmi=model.fmiVersion, platforms=platforms, co_simulation=bool(model.coSimulation),
                      temperature_points=[(p.name, p.unit) for p in points], fmpy=version("fmpy"))), flush=True)
if not points:
    raise RuntimeError("No zone temperature output")
# 用 900 秒输出间隔观察四次推进后的室温。
point = points[0]
result = simulate_fmu(str(args.fmu), validate=False, start_time=86400, stop_time=90000,
                      output_interval=900, output=[point.name])
print(json.dumps(dict(point=point.name, unit=point.unit,
                      samples=[[float(row["time"]), float(row[point.name])] for row in result])))
