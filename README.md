# Agent를 이용한 기업 정보 요약 서비스 구현하기

<p align="left">
    <a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fkyopark2014%2Focean-agent&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com"/></a>
    <img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green">
</p>

여기에서는 기업의 정보를 PDF 문서로 받아서 원하는 목차로 정리하는 업무지원 프로세스에 대해 설명합니다. PDF 문서를 활용하기 위해서는 문서의 주제와 생성일에 대한 정보를 추출해 문서 chunk의 metadata로 등록하는 절차가 필요합니다. 또한 이때의 메타정보를 기준으로 문서를 검색하기 위해서는 RAG의 지식저장소가 필요합니다. 그리고 얻어진 문서의 내용을 사용자가 원하는 포맷으로 보여줄수 있도록 markdown로 정리하고 html로 공유할 수 있어야 합니다. 여기서는 PDF에서 정보의 추출 및 메타정보로 등록, RAG를 이용한 문서의 검색, markdown형태로 문서를 공유하는 일련의 과정을 설명합니다.


## OpenSearch를 이용한 RAG의 구현

LangChain의 [OpenSearchVectorSearch](https://api.python.langchain.com/en/latest/vectorstores/langchain_community.vectorstores.opensearch_vector_search.OpenSearchVectorSearch.html)을 이용하여 지식저장소인 Amazon OpenSearch와 연결합니다. 이후 계층적 chunking을 이용하여 관련된 문서를 조회합니다. 

```python
def get_answer_using_opensearch(chat, text, connectionId, requestId):    
    global reference_docs
    
    msg = ""
    top_k = 4
    relevant_docs = []
    
    bedrock_embedding = get_embedding()
       
    vectorstore_opensearch = OpenSearchVectorSearch(
        index_name = index_name,
        is_aoss = False,
        ef_search = 1024, # 512(default)
        m=48,
        #engine="faiss",  # default: nmslib
        embedding_function = bedrock_embedding,
        opensearch_url=opensearch_url,
        http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
    )  
    
    if enalbeParentDocumentRetrival == 'true': # parent/child chunking
        relevant_documents = get_documents_from_opensearch(vectorstore_opensearch, text, top_k)
                        
        for i, document in enumerate(relevant_documents):
            parent_doc_id = document[0].metadata['parent_doc_id']
            doc_level = document[0].metadata['doc_level']
            
            excerpt, name, url = get_parent_content(parent_doc_id) # use pareant document
            
            relevant_docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'url': url,
                        'doc_level': doc_level,
                        'from': 'vector'
                    },
                )
            )
    else: 
        relevant_documents = vectorstore_opensearch.similarity_search_with_score(
            query = text,
            k = top_k,
        )
        
        for i, document in enumerate(relevant_documents):
            name = document[0].metadata['name']
            url = document[0].metadata['url']
            content = document[0].page_content
                   
            relevant_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'name': name,
                        'url': url,
                        'from': 'vector'
                    },
                )
            )

    filtered_docs = grade_documents(text, relevant_docs) # grading
    
    filtered_docs = check_duplication(filtered_docs) # check duplication
            
    relevant_context = ""
    for i, document in enumerate(filtered_docs):
        if document.page_content:
            content = document.page_content
            
        relevant_context = relevant_context + content + "\n\n"
        
    msg = query_using_RAG_context(connectionId, requestId, chat, relevant_context, text)
    
    reference_docs += filtered_docs
           
    return msg
```

조회한 문서의 관련도는 아래와 같이 LLM을 이용하여 grading을 수행합니다. 문서의 관련도 평가는 [LLM으로 RAG Grading 활용하기](https://github.com/kyopark2014/korean-chatbot-using-amazon-bedrock/blob/main/RAG-grading.md)를 참조합니다.

```python
def grade_documents(question, documents):
    print("###### grade_documents ######")
    
    filtered_docs = []
    if multi_region == 'enable':  # parallel processing
        print("start grading...")
        filtered_docs = grade_documents_using_parallel_processing(question, documents)

    else:
        # Score each doc    
        chat = get_chat()
        retrieval_grader = get_retrieval_grader(chat)
        for i, doc in enumerate(documents):            
            score = retrieval_grader.invoke({"question": question, "document": doc.page_content})
            
            grade = score.binary_score

            # Document relevant
            if grade.lower() == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            # Document not relevant
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                continue
    
    return filtered_docs
```



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

### Metadata 추출 

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
            
        if not info['parsed'] == None:
            parsed_info = info['parsed']
            subject_company = parsed_info.subject_company
            rating_date = parsed_info.rating_date                            
            break
    return subject_company, rating_date        
```

### 이미지의 Header / Footer 제거

[lambda-document-manager - lambda_function.py](./lambda-document-manager/lambda_function.py)에서는 pdf_profile을 참조하여 이미지에서 텍스트 추출시에 header와 footer를 제거합니다. header와 footer의 위치는 pdf에 맞게 조정합니다.

```python
pdf_profile = 'ocean'

def store_image_for_opensearch(key, page, subject_company, rating_date):
    image_obj = s3_client.get_object(Bucket=s3_bucket, Key=key)
                        
    image_content = image_obj['Body'].read()
    img = Image.open(BytesIO(image_content))
                        
    width, height = img.size 
    print(f"(original) width: {width}, height: {height}, size: {width*height}")
    
    pos = key.rfind('/')
    prefix = key[pos+1:pos+5]
    print('img_prefix: ', prefix)    
    if pdf_profile=='ocean' and prefix == "img_":
        area = (0, 175, width, height-175)
        img = img.crop(area)
            
        width, height = img.size 
        print(f"(croped) width: {width}, height: {height}, size: {width*height}")
```

### 문서 생성

문서의 목차와 이에 따른 작성과정은 [plan and execute 패턴](https://github.com/kyopark2014/writing-agent)을 이용합니다. 이 패턴은 이전 작성된 문서를 참고할 수 있어서 문장의 중복 및 자연스러운 연결을 위해 유용합니다. 문서의 검색과 생성은 workflow를 이용해 구성합니다.

```python
def buildWorkflow():
    workflow = StateGraph(State)
    workflow.add_node("retrieve", parallel_retriever)
        
    workflow.add_node("generate", generate_node)
    
    workflow.add_edge(START, "retrieve")
    
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()
```

검색은 parallel_retriever를 이용합니다.

```python
def parallel_retriever(state: State):
    print('###### parallel_retriever ######')
    subject_company = state["subject_company"]    
    planning_steps = state["planning_steps"]
    print(f"subject_company: {subject_company}, planning_steps: {planning_steps}")
    
    relevant_contexts = []    
    
    sub_quries = state["sub_quries"]
    
    for i, step in enumerate(planning_steps):
        print(f"{i}: {step}")

        context = ""        
        for q in sub_quries[i]:
            docs = retrieve(q, subject_company)
            
            print(f"---> q: {sub_quries[i]}, docs: {docs}")
            
            for doc in docs:            
                context += doc.page_content
            
        relevant_contexts.append(context)
        
    return {
        "subject_company": subject_company,
        "planning_steps": planning_steps,
        "relevant_contexts": relevant_contexts
    }
```

이때 RAG 검색은 아래와 같습니다.

```python
def retrieve(query: str, subject_company: str):
    print(f'###### retrieve: {query} ######')
    global reference_docs
    
    top_k = 4
    docs = []
    
    bedrock_embedding = get_embedding()
       
    vectorstore_opensearch = OpenSearchVectorSearch(
        index_name = index_name,
        is_aoss = False,
        ef_search = 1024, # 512(default)
        m=48,
        #engine="faiss",  # default: nmslib
        embedding_function = bedrock_embedding,
        opensearch_url=opensearch_url,
        http_auth=(opensearch_account, opensearch_passwd), # http_auth=awsauth,
    )  
    
    if enalbeParentDocumentRetrival == 'true': # parent/child chunking
        relevant_documents = get_documents_from_opensearch_for_subject_company(vectorstore_opensearch, query, top_k, subject_company)
                        
        for i, document in enumerate(relevant_documents):
            parent_doc_id = document[0].metadata['parent_doc_id']
            doc_level = document[0].metadata['doc_level']

            excerpt, name, url = get_parent_content(parent_doc_id) # use pareant document
            
            docs.append(
                Document(
                    page_content=excerpt,
                    metadata={
                        'name': name,
                        'url': url,
                        'doc_level': doc_level,
                        'from': 'vector'
                    },
                )
            )
    else: 
        relevant_documents = vectorstore_opensearch.similarity_search_with_score(
            query = query,
            k = top_k,  
            pre_filter={
                "doc_level": {"$eq": "child"},
                "subject_company": {"$eq": subject_company}
            }
        )
    
        for i, document in enumerate(relevant_documents):
            name = document[0].metadata['name']
            url = document[0].metadata['url']
            content = document[0].page_content
                   
            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        'name': name,
                        'url': url,
                        'from': 'vector'
                    },
                )
            )
    
    filtered_docs = grade_documents(query, docs) # grading
    
    filtered_docs = check_duplication(filtered_docs) # check duplication
            
    reference_docs += filtered_docs # add to reference
    
    return filtered_docs
```

문서의 생성은 아래와 같이 주제에 대한 보고서 지시사항, 보고서 단계, 이미 작성한 텍스트, 참고 문서를 활용해 생성합니다. 

```python
def generate_node(state: State):    
    print('###### generate_node ######')
    write_template = (
        "당신은 기업에 대한 보고서를 작성하는 훌륭한 글쓰기 도우미입니다."
        "아래와 같이 원본 보고서 지시사항과 계획한 보고서 단계를 제공하겠습니다."
        "또한 제가 이미 작성한 텍스트를 제공합니다."
        
        "보고서 지시사항:"
        "<instruction>"
        "{instruction}"
        "</instruction>"
        
        "보고서 단계:"
        "<plan>"
        "{plan}"
        "</plan>"
        
        "이미 작성한 텍스트:"
        "<text>"
        "{text}"
        "</text>"
        
        "참고 문서"
        "<context>"        
        "{context}"
        "</context>"        
        
        "보고서 지시 사항, 보고서 단계, 이미 작성된 텍스트, 참고 문서를 참조하여 다음 단계을 계속 작성합니다."
        "기업에 대한 구체적인 정보는 받드시 참고 문서를 이용해 작성하고, 모르는 부분은 포함하지 않습니다."
        
        "다음 단계:"
        "<step>"
        "{STEP}"
        "</step>"
                
        "보고서의 내용이 끊어지지 않고 잘 이해되도록 하나의 문단을 충분히 길게 작성합니다."
        "필요하다면 앞에 작은 부제를 추가할 수 있습니다."
        "이미 작성된 텍스트를 반복하지 말고 작성한 문단만 출력하세요."                
        "Markdown 포맷으로 서식을 작성하세요."
        "최종 결과에 <result> tag를 붙여주세요."
    )
    
    write_prompt = ChatPromptTemplate.from_messages([
        ("human", write_template)
    ])
    
    instruction = f"{state['subject_company']} 회사에 대해 소개해 주세요."
    planning_steps = state["planning_steps"]
    text = ""
    drafts = []
    
    for i, step in enumerate(planning_steps):
        context = state["relevant_contexts"][i]
        
        chat = get_chat()                       
        write_chain = write_prompt | chat            
        try: 
            result = write_chain.invoke({
                "instruction": instruction,
                "plan": planning_steps,
                "text": text,
                "context": context,
                "STEP": step
            })

            output = result.content
            draft = output[output.find('<result>')+8:len(output)-9] # remove <result> tag    
                
            if draft.find('#')!=-1 and draft.find('#')!=0:
                draft = draft[draft.find('#'):]
                    
            text += draft + '\n\n'
            drafts.append(draft)
                
        except Exception:
            err_msg = traceback.format_exc()
            print('error message: ', err_msg)                        
            raise Exception ("Not able to request to LLM")

    return {
        "drafts": drafts
    }
```



## 직접 실습 해보기

### 사전 준비 사항

이 솔루션을 사용하기 위해서는 사전에 아래와 같은 준비가 되어야 합니다.

- [AWS Account 생성](https://repost.aws/ko/knowledge-center/create-and-activate-aws-account)에 따라 계정을 준비합니다.

### CDK를 이용한 인프라 설치

본 실습에서는 us-west-2 리전을 사용합니다. [인프라 설치](./deployment.md)에 따라 CDK로 인프라 설치를 진행합니다. 

## 실행결과

### RAG 기본 동작

### Agent RAG 기본 동작

채팅 메뉴에서 "Agentic RAG"를 선택한 후에 "Suzano는 어떤 회사이지?"라고 입력하면 아래와 같이 RAG와 웹검색을 통해 얻어진 정보와 관련 문서를 확인할 수 있습니다.

![noname](https://github.com/user-attachments/assets/26af0bc3-fbee-4f8d-971e-be899cc4124b)

### 복잡 질문에 대한 동작 

"RAG (OpenSearch)" 메뉴로 진입해서 "Suzano와 Delta Corp Shipping을 비교해주세요."라고 입력합니다. RAG는 사용자의 질문에 2가지 검색이 필요한 사항들이 있음에도 질문을 그대로 검색합니다. 따라서, 아래와 같이 일부 자료만 검색이 될 수 있습니다.

![image](https://github.com/user-attachments/assets/078333cd-4b81-4e12-ad8e-593578343c0a)

이제, "Delta Corp Shipping에 대해 설명해주세요."라고 입력하면 아래와 같이 해당 회사에 대한 정보를 RAG가 충분히 가지고 있음을 알 수 있습니다. 

![noname](https://github.com/user-attachments/assets/151af07e-6894-435e-a804-4926bf6bd8c5)

이제 "Agentic RAG" 메뉴로 이동하여 "Suzano와 Delta Corp Shipping을 비교해주세요."라고 다시 입력합니다. RAG에서는 2가지 검색이 필요한 질문을 잘 처리하지 못하였지만, Agentic RAG는 아래와 같이 두 회사를 잘 비교하고 있습니다.

![noname](https://github.com/user-attachments/assets/fd540892-8cbe-4cba-9886-e7d45e48bc75)

이때의 LangSmith의 로그를 확인하면, 아래와 같이 OpenSearch로 "Suzano"와 "Delta Corp Shipping"을 각각 조회하여 얻은 결과를 가지고 최종 답변을 얻은것을 알 수 있습니다. 이와같이 [query decomposition](https://github.com/kyopark2014/rag-with-reflection)을 이용하면, RAG 검색의 결과를 향상 시킬 수 있습니다.

![noname](https://github.com/user-attachments/assets/74de5172-acc8-440e-8f11-2ce0385d4099)

### 기업 보고서 작성

메뉴에서 "Ocean Agent"를 선택한 후에 "Suzano SA"로 회사의 이름만을 입력합니다. 미리보기 링크를 누르면 html로 문서 내용을 공유 할 수 있습니다. 다운로드 링크를 누르면 markdown 형태로 파일을 다운로드 해서 github 또는 메일로 공유할 수 있습니다.

<img src="./contents/Suzano_SA.png" width="700">

## 결론

OpenSearch를 활용하여 RAG를 생성하고, 기업 정보를 저장하여 분석할 수 있었습니다. 또한 Agentic RAG를 구성하여 RAG뿐 아니라 일반 대화와 웹검색을 구현할 수 있습니다. 여기서는 인프라를 효율적으로 관리하기 위하여 AWS CDK로 OpenSearch를 설치하고 유지보수 및 변화하는 트래픽 처리에 유용한 서버리스 서비스 중심으로 시스템을 구성하였습니다. 


## 리소스 정리하기 

더이상 인프라를 사용하지 않는 경우에 아래처럼 모든 리소스를 삭제할 수 있습니다. 

1) [API Gateway Console](https://us-west-2.console.aws.amazon.com/apigateway/main/apis?region=us-west-2)로 접속하여 "api-chatbot-for-ocean-agent", "api-ocean-agent"을 삭제합니다.

2) [Cloud9 Console](https://us-west-2.console.aws.amazon.com/cloud9control/home?region=us-west-2#/)에 접속하여 아래의 명령어로 전체 삭제를 합니다.

```text
cd ~/environment/ocean-agent/cdk-ocean-agent/ && cdk destroy --all
```
