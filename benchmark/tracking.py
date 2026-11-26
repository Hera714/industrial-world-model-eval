"""总说明：本地 MLflow 可选追踪；每个 benchmark run 对应一个 MLflow run，记录成功及失败。"""
from pathlib import Path
import json, math
import yaml


class LocalTracking:
    """显式使用 MlflowClient，避免批量进程间共享全局 active run。"""
    def __init__(self, config_path, artifacts):
        from mlflow.tracking import MlflowClient
        self.artifacts = artifacts
        root = artifacts.root
        cfg = yaml.safe_load(Path(config_path).read_text())
        if cfg.get('backend') != 'local_file':
            raise ValueError('tracking.backend: expected local_file')
        directory = (root/cfg['directory']).resolve()
        if not directory.is_relative_to(root):
            raise ValueError('tracking.directory must stay inside project')
        self.uri = directory.as_uri()
        self.client = MlflowClient(tracking_uri=self.uri)
        experiment = self.client.get_experiment_by_name(cfg['experiment'])
        eid = experiment.experiment_id if experiment else self.client.create_experiment(cfg['experiment'])
        run = self.client.create_run(eid, tags={'mlflow.runName': artifacts.run_id,
                    'benchmark.run_id': artifacts.run_id, 'benchmark.directory': str(artifacts.path)})
        self.run_id = run.info.run_id
        artifacts._json('tracking.json', dict(tracking_uri=self.uri, experiment_id=eid, mlflow_run_id=self.run_id))
        try:
            self._params(artifacts.config)
            versions = json.loads((artifacts.path/'versions.json').read_text())
            self.client.set_tag(self.run_id,'benchmark.commit',str(versions.get('commit')))
        except Exception:
            self.client.set_terminated(self.run_id,status='FAILED')
            raise

    def _params(self, config):
        """展开配置为可搜索参数；完整 YAML 仍作为产物保存。"""
        def visit(value, key='config'):
            if isinstance(value, dict):
                for name, child in value.items():visit(child,key+'.'+str(name))
            elif isinstance(value,list) and any(isinstance(x,dict) for x in value):
                for index, child in enumerate(value):visit(child,key+f'.{index}')
            else:
                self.client.log_param(self.run_id,key,json.dumps(value,ensure_ascii=False)[:5500])
        visit(config)

    def finish(self):
        """本地状态落盘后同步指标与索引产物；大型轨迹留在原目录，路径与哈希均可追溯。"""
        art = self.artifacts
        status = art.state['status']
        try:
            metrics = json.loads((art.path/'metrics.json').read_text()).get('metrics') or {}
            for group in ('open_loop','closed_loop'):
                for index,row in enumerate(metrics.get(group,[])):
                    prefix=f'{group}.{index}'
                    for name,value in row.items():
                        if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):
                            self.client.log_metric(self.run_id,prefix+'.'+name,value)
                        elif name in ('mae','rmse'):
                            for variable,number in zip(metrics.get('variables',[]),value):
                                if isinstance(number,(int,float)) and math.isfinite(number):
                                    self.client.log_metric(self.run_id,prefix+'.'+name+'.'+variable,number)
            for name in ('config.yaml','versions.json','input_hashes.json','metrics.json','status.json','failure.json','context.json','artifacts.json','report.md','tracking.json'):
                path=art.path/name
                if path.is_file():self.client.log_artifact(self.run_id,str(path))
            self.client.set_tag(self.run_id,'benchmark.status',status)
            if (art.path/'failure.json').exists():
                failure=json.loads((art.path/'failure.json').read_text())
                self.client.set_tag(self.run_id,'benchmark.error_type',failure['error_type'])
        finally:
            self.client.set_terminated(self.run_id,status='FINISHED' if status=='completed' else 'FAILED')
