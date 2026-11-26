"""总说明：管理每次 benchmark 的独立产物目录，原子保存状态、结果与失败上下文，禁止静默覆盖。"""
from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import traceback
import uuid
import numpy as np
import yaml
from benchmark.config import checked_path, validate_config


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def serializable(value):
    """使用标准 JSON；NaN/Inf 显式标记，避免把失败值悄悄改为零或删除。"""
    if isinstance(value, np.ndarray):
        return serializable(value.tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': str(value)}
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def atomic_write(path, text):
    """同目录临时文件落盘后替换；读者不会看到半份 JSON/YAML。"""
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temp.open('w', encoding='utf-8', newline='') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def json_text(value):
    return json.dumps(serializable(value), ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def version_info(root):
    """记录代码和依赖版本，不采集完整环境变量、代理凭据或远端仓库地址。"""
    def git(*args):
        try:
            result = subprocess.run(['git', '-C', str(root), *args], capture_output=True,
                                    text=True, timeout=10, check=True)
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    packages = {}
    for name in ['numpy', 'pandas', 'torch', 'PyYAML', 'gymnasium', 'fmpy']:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    dirty = git('status', '--porcelain', '--untracked-files=normal')
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                packages=packages, commit=git('rev-parse', 'HEAD'),
                git_dirty=None if dirty is None else bool(dirty),
                git_changes=None if dirty is None else dirty.splitlines())


class RunArtifacts:
    """一个实例对应一个 run；上下文内异常会记为 failed 并继续抛出，正常退出完成产物校验。"""
    def __init__(self, config, root=None, tracking_config=None):
        self.root = Path(root or Path(__file__).resolve().parents[1]).resolve()
        self.config = validate_config(config, self.root)
        parent = checked_path(self.root, self.config['output']['directory'], 'output.directory')
        self.run_id = (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '_'
                       + self.config['output']['run_name'] + '_' + uuid.uuid4().hex[:12])
        self.path = parent / self.run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.tracker = None
        self.context = {}
        self.rows = 0
        self.fields = None
        self.has_metrics = False
        self.state = dict(run_id=self.run_id, status='running', started_at=utc_now(),
                          finished_at=None, trajectory_rows=0)
        self._json('status.json', self.state)
        try:
            snapshot = yaml.safe_dump(self.config, sort_keys=False, allow_unicode=True)
            atomic_write(self.path/'config.yaml', snapshot)
            self._json('versions.json', version_info(self.root))
            self._json('input_hashes.json', self._input_hashes(snapshot))
            self._json('metrics.json', {'status': 'pending', 'metrics': None})
            for name in ['trajectories.csv', 'events.jsonl', 'run.log']:
                (self.path/name).touch()
            self.log('run_started')
            if tracking_config:
                from benchmark.tracking import LocalTracking
                self.tracker = LocalTracking(tracking_config, self)
        except BaseException as error:
            self.fail(error)
            raise

    def _json(self, name, value):
        atomic_write(self.path/name, json_text(value))

    def _input_hashes(self, snapshot):
        """记录实际输入哈希，包括清单内的数据与上游构建资产；不复制大文件。"""
        references = [self.config['dataset']['manifest'], self.config['env']['assets']]
        references += [s['checkpoint'] for s in self.config['model']['instances']]
        ledger = json.loads((self.root/self.config['dataset']['manifest']['path']).read_text())
        references += [entry['file'] for entry in ledger['episodes']]
        assets = json.loads((self.root/self.config['env']['assets']['path']).read_text())
        references += [dict(path=p, sha256=h) for p, h in assets['sha256'].items()]
        files = {}
        for spec in references:
            actual = hashlib.sha256((self.root/spec['path']).read_bytes()).hexdigest()
            if actual != spec['sha256']:
                raise ValueError('Input changed during run initialization: '+spec['path'])
            files[spec['path']] = actual
        return dict(config_sha256=hashlib.sha256(snapshot.encode()).hexdigest(), files=files)

    def _running(self):
        if self.state['status'] != 'running':
            raise RuntimeError('Run already finalized')

    def set_context(self, **context):
        """由 Runner 更新 task/model/episode/origin/step；崩溃时保存最后上下文。"""
        self._running()
        self.context.update(serializable(context))
        self._json('context.json', self.context)

    def log(self, event, level='INFO', **details):
        """逐条关闭并刷出日志，避免异常退出丢失缓冲中的记录。"""
        self._running()
        record = dict(time=utc_now(), level=level, event=event, context=self.context.copy(), **details)
        with (self.path/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(serializable(record), ensure_ascii=False, allow_nan=False)+'\n')
        with (self.path/'run.log').open('a', encoding='utf-8') as stream:
            stream.write(f"{record['time']} {level} {event} {json.dumps(serializable(details), ensure_ascii=False)}\n")

    def append_trajectory(self, row):
        """首条记录冻结列集合；后续缺列/多列报错，失败前已写行继续保留。"""
        self._running()
        if not isinstance(row, dict) or not row:
            raise ValueError('Trajectory row must be a nonempty mapping')
        if self.fields is not None and set(row) != set(self.fields):
            raise ValueError('Trajectory columns mismatch')
        first = self.fields is None
        fields = list(row) if first else self.fields
        values = {k: (json.dumps(serializable(v), ensure_ascii=False, allow_nan=False)
                      if isinstance(v, (dict, list, tuple, np.ndarray)) else v) for k, v in row.items()}
        with (self.path/'trajectories.csv').open('a', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            if first:
                writer.writeheader()
            writer.writerow(values)
        self.fields = fields
        self.rows += 1

    def write_metrics(self, metrics):
        """指标保留失败数、有效率等调用者给出的字段；不会过滤失败样本。"""
        self._running()
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError('Metrics must be a nonempty mapping')
        self._json('metrics.json', {'status': 'available', 'metrics': metrics})
        self.has_metrics = True

    def write_predictions(self, **arrays):
        """保存可选原始预测 NPZ，拒绝 object 数组，读取无需开启 pickle。"""
        self._running()
        if not arrays or any(np.asarray(a).dtype.hasobject for a in arrays.values()):
            raise ValueError('Predictions require non-object arrays')
        if (self.path/'predictions.npz').exists():
            raise FileExistsError('Predictions already written')
        temp = self.path/('.predictions.'+uuid.uuid4().hex+'.npz')
        try:
            np.savez_compressed(temp, **arrays)
            os.replace(temp, self.path/'predictions.npz')
        finally:
            temp.unlink(missing_ok=True)

    def write_report(self, markdown):
        self._running()
        atomic_write(self.path/'report.md', markdown)

    def _manifest(self):
        """终态清单排除自身及 status，避免循环哈希；状态最后写入。"""
        self._json('artifacts.json', {p.name: dict(bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest())
                    for p in sorted(self.path.iterdir()) if p.is_file()
                    and p.name not in ('artifacts.json', 'status.json')})

    def finish(self):
        self._running()
        if not self.has_metrics or not self.rows:
            raise RuntimeError('Cannot complete run without metrics and trajectory rows')
        self.log('run_completed')
        self._manifest()
        self.state.update(status='completed', finished_at=utc_now(), trajectory_rows=self.rows)
        self._json('status.json', self.state)

    def fail(self, error):
        if self.state['status'] != 'running':
            return
        failure = dict(time=utc_now(), error_type=type(error).__name__, message=str(error),
                       traceback=''.join(traceback.format_exception(type(error), error, error.__traceback__)),
                       context=self.context, trajectory_rows=self.rows)
        self._json('failure.json', failure)
        self.log('run_failed', level='ERROR', error_type=failure['error_type'], message=failure['message'])
        self._manifest()
        self.state.update(status='failed', finished_at=utc_now(), trajectory_rows=self.rows)
        self._json('status.json', self.state)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, error, tb):
        try:
            if error is not None:
                self.fail(error)
            elif self.state['status'] == 'running':
                try:
                    self.finish()
                except BaseException as failure:
                    self.fail(failure)
                    raise
        finally:
            if self.tracker:
                try:
                    self.tracker.finish()
                except Exception as tracking_error:
                    self._json('tracking_error.json', dict(error_type=type(tracking_error).__name__, message=str(tracking_error)))
                    self._manifest()
                    import warnings
                    warnings.warn('MLflow 同步失败，详见 tracking_error.json: '+str(tracking_error))
        return False
