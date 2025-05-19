class HistoryManager:
    def __init__(self, template=[]):
        # 모든 대화 히스토리를 저장할 딕셔너리
        self.histories = {}
        # 기본 템플릿 히스토리
        self.default_template = template.copy()
    
    def get_all_history(self):
        return self.histories
    
    def get_history(self, conversation_id):
        """
        주어진 id에 해당하는 히스토리를 반환합니다.
        해당 id가 없으면 기본 템플릿으로 초기화합니다.
        """
        if conversation_id not in self.histories:
            # 새로운 id면 기본 템플릿으로 초기화
            self.histories[conversation_id] = self.default_template.copy()
        return self.histories[conversation_id]
    
    def get_filtered_history(self, conversation_id, filter_role=None, filter_keyword=None, filter_image_url=False):
        """
        주어진 id에 해당하는 히스토리를 반환합니다.
        해당 id가 없으면 기본 템플릿으로 초기화합니다.

        :param conversation_id: 대화 식별자
        :param filter_role: 이 role을 가진 메시지는 결과에서 제외
        :param filter_keyword: content 안에 이 키워드가 포함된 메시지는 결과에서 제외
        :param filter_image_url: True이면 content 안에 {"type": "image_url"}이 하나라도 있으면 결과에서 제외
        :return: 필터링 결과를 반영한 history 리스트
        """

        # 만약 conversation_id가 없다면 기본 템플릿으로 초기화
        if conversation_id not in self.histories:
            self.histories[conversation_id] = self.default_template.copy()

        # 전체 히스토리
        full_history = self.histories[conversation_id]

        # 필터링된 히스토리 구성
        filtered_history = []
        for message in full_history:
            # 1) 특정 role을 가진 메시지 제외
            if filter_role is not None and message.get('role') == filter_role:
                continue

            # 2) content 내 특정 키워드가 포함된 메시지 제외
            if filter_keyword is not None:
                # message['content']가 문자열인 경우나 리스트인 경우 모두를 고려
                msg_content = message.get('content', "")
                if isinstance(msg_content, str):
                    if filter_keyword in msg_content:
                        continue
                elif isinstance(msg_content, list):
                    # 리스트 중 어떤 항목에라도 filter_keyword가 있으면 제외
                    if any(filter_keyword in str(item) for item in msg_content):
                        continue

            # 3) content 내 'image_url' 타입이 하나라도 있으면 메시지 제외
            if filter_image_url:
                msg_content = message.get('content', [])
                if isinstance(msg_content, list):
                    # content가 list일 경우: [{'type': 'text', ...}, {'type': 'image_url', ...}, ...]
                    # 여기서 'type'이 'image_url'인 요소가 하나라도 있으면 제외
                    if any(item.get('type') == 'image_url' for item in msg_content):
                        continue
                else:
                    # content가 dict나 문자열 등 다른 형태면 그대로 패스
                    pass

            # 위 조건을 모두 통과하면 필터링되지 않으므로 결과에 추가
            filtered_history.append(message)

        return filtered_history
    
    def add_message_direct(self, conversation_id, message):
        if conversation_id not in self.histories:
            self.histories[conversation_id] = []
        self.histories[conversation_id].append(message)
        return self.histories[conversation_id]
        
    def add_message(self, conversation_id, role, content):
        """
        주어진 id의 대화 히스토리에 새 메시지를 추가합니다.
        """
        # id가 없다면 기본 템플릿으로 먼저 초기화
        if conversation_id not in self.histories:
            self.histories[conversation_id] = self.default_template.copy()
        self.histories[conversation_id].append({
            "role": role,
            "content": content
        })
        return self.histories[conversation_id]
    def add_image_message(self, conversation_id, role, text_content,  image_content, image_format='gif'):
        """
        주어진 id의 대화 히스토리에 새 메시지를 추가합니다.
        """
        # id가 없다면 기본 템플릿으로 먼저 초기화
        if conversation_id not in self.histories:
            self.histories[conversation_id] = self.default_template.copy()
        self.histories[conversation_id].append({
            "role": role,
            "content":  [{
                "type": "text",
                "text": text_content
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_format};base64,{image_content}"
                }
            }
        ]})
        return self.histories[conversation_id]
    
    def delete_history_by_id(self, conversation_id):
        try:
            del self.histories[conversation_id]
            return conversation_id
        except:
            print('failed to delete',conversation_id)

    def get_last_prompt(self, conversation_id):
        return self.get_history(conversation_id)[len(self.histories[conversation_id])-1]

    def get_prompt(self, conversation_id):
        """
        API 요청 시 사용할 전체 프롬프트(대화 히스토리)를 반환합니다.
        """
        return self.get_history(conversation_id)
    
    def pop_message(self, conversation_id):
        """
        주어진 id에 해당하는 히스토리를 삭제합니다.
        """
        if conversation_id in self.histories:
            self.histories[conversation_id].pop()
        return self.histories[conversation_id]
