"""工具契约：输入和输出均为 JSON 对象，不支持 shell 指令。"""
TOOL_SCHEMAS = {
 'list_envs': {'required': [], 'allowed': []},
 'validate_config': {'required': ['config'], 'allowed': ['config']},
 'run_benchmark': {'required': ['config'], 'allowed': ['config']},
 'summarize_results': {'required': ['run_directory'], 'allowed': ['run_directory']},
}

def validate_arguments(name, arguments):
    if name not in TOOL_SCHEMAS: raise ValueError('UNKNOWN_TOOL: 未知工具')
    if not isinstance(arguments,dict): raise ValueError('INVALID_ARGUMENT: 参数必须为对象')
    schema=TOOL_SCHEMAS[name]
    missing=set(schema['required'])-set(arguments)
    extra=set(arguments)-set(schema['allowed'])
    if missing or extra: raise ValueError(f'INVALID_ARGUMENT: 缺少 {sorted(missing)}；不支持 {sorted(extra)}')

# 每个函数的 JSON Schema 用于模型工具接口；执行前仍进行严格校验。
TOOL_DEFINITIONS = [
 {'type':'function','function':{'name':'list_envs','description':'列出环境','parameters':{'type':'object','properties':{},'additionalProperties':False}}},
 {'type':'function','function':{'name':'validate_config','description':'验证配置','parameters':{'type':'object','properties':{'config':{'type':'object'}},'required':['config'],'additionalProperties':False}}},
 {'type':'function','function':{'name':'run_benchmark','description':'执行已验证评测','parameters':{'type':'object','properties':{'config':{'type':'object'}},'required':['config'],'additionalProperties':False}}},
 {'type':'function','function':{'name':'summarize_results','description':'审核运行结果','parameters':{'type':'object','properties':{'run_directory':{'type':'string'}},'required':['run_directory'],'additionalProperties':False}}},
]
