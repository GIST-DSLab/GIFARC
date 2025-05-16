import json

class ExecutionError(Exception):
    def __init__(self, error_info, target_input):
        self.error_info = error_info
        self.target_input = target_input
        # error_info를 문자열로 변환해 포함시키거나 JSON 문자열로 만들 수 있습니다.
        message = json.dumps({
            "error_info": str(error_info),
            "target_input": target_input
        })
        super().__init__(message)