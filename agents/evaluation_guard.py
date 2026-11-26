"""冻结测试集的执行闸门，禁止对不应执行的用例启动真实仿真。"""
def may_execute(case,intent,validated):
    return bool(validated and case.get('expected_action')=='generate_and_validate' and intent.get('models')==case.get('expected_models'))
