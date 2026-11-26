"""统一解析模型回复，异常携带原始文本供审计。"""
import json
from agents.config_generator import config_from_intent

class ModelResponseError(ValueError):
    def __init__(self,message,raw_response):
        super().__init__(message)
        self.raw_response=raw_response

def parse_response(raw):
    try:
        intent=json.loads(raw)
        if not isinstance(intent,dict):raise ValueError('INVALID_INTENT: 需要 JSON 对象')
        if 'error' in intent:raise ValueError('AMBIGUOUS_REQUEST: '+str(intent['error']))
        config=config_from_intent(intent)
        return intent,config
    except (ValueError,KeyError,TypeError) as error:
        raise ModelResponseError(str(error),raw) from error
