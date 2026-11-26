"""总说明：统一分类评测过程中的错误，给出可操作建议，保留原始异常类型与消息。"""
import socket
import re

def describe_error(error):
    message=str(error)
    text=message.lower()
    if isinstance(error,(TimeoutError,socket.timeout)) or 'timeout' in text or 'timed out' in text:
        code='TIMEOUT';suggestion='检查模型服务或 BOPTEST 是否可用和超时预算；推进失败后重新初始化回合。'
    elif 'env_not_allowed' in text or '非法环境' in text:
        code='INVALID_ENVIRONMENT';suggestion='使用 list_envs 查询支持的环境，或扩展并验证白名单。'
    elif 'missing required field' in text or isinstance(error,KeyError) or '需要且仅允许' in text:
        code='MISSING_FIELD';suggestion='检查错误中指出的字段，使用冻结模板补全配置。'
    elif ('nonfinite' in text or 'invalid predicted' in text or
          re.search(r'(?<![a-z0-9_])(?:nan|[+-]?inf(?:inity)?)(?![a-z0-9_])',text)):
        code='INVALID_PREDICTION';suggestion='检查输入、模型及原始失败轨迹，保留失败样本并核查数值稳定性。'
    else:
        code='EXECUTION_ERROR';suggestion='检查会话日志和原始错误，修复后重新执行。'
    result=dict(code=code,type=type(error).__name__,message=message,suggestion=suggestion)
    if hasattr(error,'raw_response'):result['raw_response']=error.raw_response
    return result
