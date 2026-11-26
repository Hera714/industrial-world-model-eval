"""总说明：让本地 LLM 根据工具观察自主选择下一工具，工具输入仍由确定性白名单保护。"""
import json
import torch
from agents.config_generator import config_from_intent

PROMPT=('你是实验工具编排器。每轮只输出 JSON 对象 {"tool":名称,"models":[检查点名]}。'
 '合法工具 list_envs、validate_config、run_benchmark、summarize_results、finish。'
 '只能运行用户明确要求且已验证的 BOPTEST 模型；依次验证、运行、审核，失败时不要声称成功。'
 '工具结果由系统提供。不得生成 shell 或访问其他环境。'
 '例如：用户要求 BOPTEST persistence，历史为空时返回 {"tool":"validate_config","models":["persistence"]}。'
 '如果 validate_config 的观察 ok 为 true，下一步返回 {"tool":"run_benchmark","models":["persistence"]}。'
 '如果最近一步 run_benchmark 的观察 ok 为 true，下一步返回 {"tool":"summarize_results"}。'
 '如果最近一步 summarize_results 的观察 ok 为 true 且 result.passed 为 true，下一步必须返回 {"tool":"finish"}，不要重新验证或运行。'
 '如果工具错误，分析错误并选择允许的工具，不要输出 unknown_tool 或 checkpoint。'
 '模型名称仅可从以下列表选：persistence, arx, mlp_seed42, mlp_seed43, mlp_seed44, gru_seed42, gru_seed43, gru_seed44, residual_mlp_seed42, residual_mlp_seed43, residual_mlp_seed44。'
 'validate_config 和 run_benchmark 的 models 必须非空且和用户所选模型一致。例如用户要求 arx 时使用 models 为 ["arx"]。')

class LocalToolDecider:
    """通过真实本地模型生成动作，不从测试集答案推导工具。"""
    def __init__(self,generator):self.generator=generator
    def __call__(self,request,history):
        generator=self.generator
        # 审核结果可能包含数百条检查：向模型传递决策所需摘要，完整结果仍留在轨迹。
        compact=[]
        for item in history:
            observation=item['observation']
            result=observation.get('result') or {}
            if item['tool']=='summarize_results' and isinstance(result,dict):
                result={'passed':result.get('passed')}
            compact.append(dict(tool=item['tool'],observation=dict(ok=observation.get('ok'),
                result=result,error=observation.get('error'))))
        messages=[{'role':'system','content':PROMPT},{'role':'user','content':json.dumps(dict(request=request,history=compact),ensure_ascii=False)}]
        text=generator.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=generator.tokenizer([text],return_tensors='pt').to(generator.device)
        with torch.inference_mode():output=generator.model.generate(**inputs,max_new_tokens=256,do_sample=False)
        raw=generator.tokenizer.decode(output[0][inputs.input_ids.shape[1]:],skip_special_tokens=True)
        action=json.loads(raw);name=action['tool'];models=action.get('models',[])
        if name in ('validate_config','run_benchmark'):
            try:
                config=config_from_intent(dict(environment='boptest',models=models))
            except ValueError as error:
                raise ValueError(str(error)+'; model_response='+raw) from error
            return dict(tool=name,arguments=dict(config=config),raw_response=raw)
        if name=='summarize_results':
            runs=[x['observation']['result']['run_directory'] for x in history if x['tool']=='run_benchmark' and x['observation'].get('ok')]
            if not runs:raise ValueError('缺少成功的 run')
            return dict(tool=name,arguments=dict(run_directory=runs[-1]),raw_response=raw)
        return dict(tool=name,arguments={},raw_response=raw)
