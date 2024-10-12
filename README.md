# Agent를 이용한 정보 조회 서비스

<p align="left">
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkyopark2014%2Focean-agent&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com"/></a>
    <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
</p>


## Agentic RAG

Agent로 RAG가 포함된 workflow를 아래와 같이 구성합니다. Tool에는 시간(get_current_time), 도서(get_book_list), 날씨(get_weather_info)와 같은 기본 기능뿐 아니라, 웹검색(search_by_tavily)과 기업정보 검색(search_by_opensearch)을 위한 기능을 포함하고 있습니다. 

```python
class State(TypedDict):
    messages: Annotated[list, add_messages]

tools = [get_current_time, get_book_list, get_weather_info, search_by_tavily, search_by_opensearch]
tool_node = ToolNode(tools)

def buildChatAgent():
    workflow = StateGraph(State)

    workflow.add_node("agent", call_model)
    workflow.add_node("action", tool_node)
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "action",
            "end": END,
        },
    )
    workflow.add_edge("action", "agent")

    return workflow.compile()
```

call_model 노드에서는 agent의 이름롸 역할을 지정하고, 이전 대화와 Tool등으로 부터 얻어진 정보를 활용하여 적절한 답변을 수행합니다.

```python
def call_model(state: State):
    print("###### call_model ######")
    
    if isKorean(state["messages"][0].content)==True:
        system = (
            "당신의 이름은 서연이고, 질문에 친근한 방식으로 대답하도록 설계된 대화형 AI입니다."
            "상황에 맞는 구체적인 세부 정보를 충분히 제공합니다."
            "모르는 질문을 받으면 솔직히 모른다고 말합니다."
            "최종 답변에는 조사한 내용을 반드시 포함합니다."
        )
    else: 
        system = (            
            "You are a conversational AI designed to answer in a friendly way to a question."
            "If you don't know the answer, just say that you don't know, don't try to make up an answer."
            "You will be acting as a thoughtful advisor."    
        )
        
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    chain = prompt | model
        
    response = chain.invoke(state["messages"])

    return {"messages": [response]}
```

## 문서 전처리

특정 페이지의 표에는 "Subject company"와 "Rating date"로 해당 문서의 대상과 생성일을 확인할 수 있습니다.

<img width="830" alt="image" src="https://github.com/user-attachments/assets/3ef0a261-678e-4af2-b58f-8c0eb9d442ec">

문서를 Amazon S3에 올릴때 발생하는 put event를 이용하여 문서를 읽어올때 특정 페이지의 정보를 이용해 company와 date를 확인합니다.

[lambda-document-manager / lambda_function.py](./lambda-document-manager/lambda_function.py)의 아래 코드를 참조합니다. 여기서는 문장에서 [Structured Output](https://github.com/kyopark2014/langgraph-agent/blob/main/structured-output.md)을 이용하여 subject_company, rating_date을 추출합니다.

```python
def get_profile_of_doc(content: str):
    """Provide profile of document."""
    
    class Profile(BaseModel):
        subject_company: str = Field(description="The value of 'Subject company'")
        rating_date: str = Field(description="The value of 'Rating data'")
    
    subject_company = rating_date = ""
    for attempt in range(5):
        chat = get_chat()
        structured_llm = chat.with_structured_output(Profile, include_raw=True)
    
        info = structured_llm.invoke(content)
        print(f'attempt: {attempt}, info: {info}')
            
        if not info['parsed'] == None:
            parsed_info = info['parsed']
            subject_company = parsed_info.subject_company
            rating_date = parsed_info.rating_date
                            
            print('subject_company: ', subject_company)            
            print('rating_date: ', rating_date)
            break
    return subject_company, rating_date        
```

## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)에 따라 계정을 준비합니다.

### CDK를 이용한 인프라 설치

본 실습에서는 us-west-2 리전을 사용합니다. [인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 

## 실행결과

채팅 메뉴에서 "Agentic RAG"를 선택한 후에 "Suzano는 어떤 회사이지?"라고 입력하면 아래와 같이 RAG와 웹검색을 통해 얻어진 정보와 관련 문서를 확인할 수 있습니다.


![noname](https://github.com/user-attachments/assets/26af0bc3-fbee-4f8d-971e-be899cc4124b)




## 결론

OpenSearch를 활용하여 RAG를 생성하고, 기업 정보를 저장하여 분석할 수 있었습니다. 또한 Agentic RAG를 구성하여 RAG뿐 아니라 일반 대화와 웹검색을 구현할 수 있습니다. 여기서는 인프라를 효율적으로 관리하기 위하여 AWS CDK로 OpenSearch를 설치하고 유지보수 및 변화하는 트래픽 처리에 유용한 서버리스 서비스중심으로 시스템을 구성하였습니다. 


## 리소스 정리하기 

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. 

1) [API Gateway Console](https://us-west-2.console.aws.amazon.com/apigateway/main/apis?region=us-west-2)로 접속하여 "api-chatbot-for-ocean-agent", "api-ocean-agent"을 삭제합니다.

2) [Cloud9 Console](https://us-west-2.console.aws.amazon.com/cloud9control/home?region=us-west-2#/)에 접속하여 아래의 명령어로 전체 삭제를 합니다.

```text
cd ~/environment/langgraph-agent/cdk-langgraph-agent/ && cdk destroy --all
```
