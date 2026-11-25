"""总说明：BOPTEST 温度设定值覆盖接口，记录实际动作与原生累计能耗；释放时交还内置控制。"""
import json
from urllib.request import Request,build_opener,ProxyHandler
import numpy as np
from gymnasium import spaces
from simulators.boptest_adapter import BoptestEnv


class BoptestControlEnv(BoptestEnv):
    """动作单位为 °C，服务端为 K；reset 后每次 step 明确覆盖或释放，不重复推进失败请求。"""
    def __init__(self,config):
        super().__init__(config)
        bounds=config['action']['experiment_bounds_c']
        self.action_space=spaces.Box(np.array([bounds[0]],np.float32),np.array([bounds[1]],np.float32),dtype=np.float32)
        self.last_energy=0.; self.area=192.

    def request(self,method,path,data=None):
        """本地 HTTP 不走系统代理；HTTP 失败保留响应正文，帮助定位服务端故障。"""
        from urllib.error import HTTPError
        url=self.config['url'].rstrip('/')+path
        opener=build_opener(ProxyHandler({})) if '127.0.0.1' in url or 'localhost' in url else build_opener()
        req=Request(url,None if data is None else json.dumps(data).encode(),{'Content-Type':'application/json'},method=method)
        try:
            with opener.open(req,timeout=180) as response:
                body=response.read()
        except HTTPError as error:
            raise RuntimeError(f'BOPTEST {method} {path}: HTTP {error.code}: {error.read().decode(errors="replace")}') from error
        if path.startswith('/stop/'): return None
        result=json.loads(body)
        if result.get('status',200)!=200: raise RuntimeError(str(result))
        return result.get('payload',result)

    def reset(self,*,seed=None,options=None):
        """seed 仅标记客户端，物理初态由时间/预热决定，不能伪装成随机物理回合。"""
        obs,info=super().reset(seed=seed,options=options)
        self.input_metadata=self.request('GET',f'/inputs/{self.testid}')
        point=self.config['action']['point']; activation=self.config['action']['activation']
        if point not in self.input_metadata or activation not in self.input_metadata: raise ValueError('Missing override points')
        meta=self.input_metadata[point]
        lo,hi=self.config['action']['experiment_bounds_c']
        if meta['Unit']!='K' or lo+273.15<meta['Minimum'] or hi+273.15>meta['Maximum']:
            raise ValueError('Setpoint metadata/bounds mismatch')
        self.initial_kpi=self.request('GET',f'/kpi/{self.testid}')
        self.last_energy=float(self.initial_kpi['ener_tot'])*self.area
        info['seed_semantics']='client only; deterministic calendar/warmup physical initialization'
        return obs,info

    def step(self,action):
        """合法设定值覆盖一个控制步；None 明确释放覆盖。非法动作不会推进环境。"""
        if self.done: raise RuntimeError('Call reset before step or after end')
        point=self.config['action']['point']; activation=self.config['action']['activation']
        if action is None:
            command={activation:0}; requested=None
        else:
            value=np.asarray(action,dtype=float)
            if value.shape!=(1,) or not np.isfinite(value).all() or not self.action_space.low[0]<=value[0]<=self.action_space.high[0]:
                raise ValueError('Expected finite legal setpoint [Celsius]')
            requested=float(value[0]); command={activation:1,point:requested+273.15}
        # 一旦发送推进请求，任何异常都要求重新 reset，不盲目重试。
        self.done=True
        points=self.request('POST',f'/advance/{self.testid}',command)
        self.steps+=1
        expected=self.config['start_time']+self.steps*self.config['step_seconds']
        if abs(points['time']-expected)>1e-6: raise RuntimeError('Unexpected simulation time')
        kpi=self.request('GET',f'/kpi/{self.testid}')
        cumulative=float(kpi['ener_tot'])*self.area
        energy=cumulative-self.last_energy
        if energy < -1e-8: raise RuntimeError('Native cumulative energy decreased')
        self.last_energy=cumulative
        applied=self.request('PUT',f'/results/{self.testid}',dict(point_names=[point],start_time=expected,final_time=expected))
        actual=float(applied[point][-1])-273.15
        if requested is not None and abs(actual-requested)>1e-5: raise RuntimeError('Override not applied')
        obs,info=self._sample(points)
        info.update(requested_action_c=requested,applied_action_c=actual,override_active=action is not None,
                    energy_kwh=max(energy,0.),native_kpi=kpi,fallback_reason='')
        self.done=self.steps>=self.config['episode_steps']
        return obs,0.,False,self.done,info

    def safe_step(self,action,reason=''):
        """规划失败/非法动作在发送前释放到内置控制；网络失败不当作可重试规划失败。"""
        if reason:
            result=self.step(None); result[-1]['fallback_reason']=reason; return result
        try:
            value=np.asarray(action,dtype=float)
            valid=value.shape==(1,) and np.isfinite(value).all() and self.action_space.low[0]<=value[0]<=self.action_space.high[0]
        except (ValueError,TypeError): valid=False
        result=self.step(action if valid else None)
        if not valid: result[-1]['fallback_reason']='invalid_planned_action'
        return result


def measurement_state(points):
    """公开可观测向量，不称为建筑完整物理状态；温度°C、热流/总电功率kW。"""
    return np.array([points['reaTZon_y']-273.15,points['reaTSup_y']-273.15,points['reaTRet_y']-273.15,
        points['reaQFloHea_y']/1000.,sum(points[key] for key in ('reaPHeaPum_y','reaPFan_y','reaPPumEmi_y'))/1000.])
