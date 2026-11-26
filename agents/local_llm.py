"""总说明：通过 Transformers 本地加载指令模型，输出仍受冻结意图与配置校验约束。"""
import json
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from agents.llm import SYSTEM_PROMPT,PROMPT_VERSION
from agents.response_parser import parse_response
from agents.intent_guard import require_explicit_boptest

class LocalGenerator:
    """模型只加载一次，固定 revision、贪心解码；记录真实返回文本和模型版本。"""
    def __init__(self,model,revision,device='cuda'):
        self.model_name=model;self.revision=revision;self.device=device
        self.tokenizer=AutoTokenizer.from_pretrained(model,revision=revision)
        self.model=AutoModelForCausalLM.from_pretrained(model,revision=revision,torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32).to(device).eval()
    def __call__(self,request,*,endpoint=None,model=None):
        require_explicit_boptest(request)
        messages=[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':request}]
        text=self.tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.tokenizer([text],return_tensors='pt').to(self.device)
        with torch.inference_mode():output=self.model.generate(**inputs,max_new_tokens=256,do_sample=False)
        raw=self.tokenizer.decode(output[0][inputs.input_ids.shape[1]:],skip_special_tokens=True)
        intent,config=parse_response(raw)
        return dict(intent=intent,config=config,metadata=dict(model=self.model_name,revision=self.revision,
                    prompt_version=PROMPT_VERSION,temperature=0,raw_response=raw,device=self.device))
