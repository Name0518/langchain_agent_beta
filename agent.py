import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

# 0. 从环境变量读取密钥（先在终端 export DEEPSEEK_API_KEY=新密钥）
load_dotenv()

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
# 1. 知识库
docs = [
    Document(page_content="William 喜欢在多伦多大学打羽毛球，并且擅长用 Python 和 C++ 进行开发。"),
    Document(page_content="项目 SHIFT 是一个在公共交通通勤时优化任务管理的效率应用。"),
]

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# 2. Tool


@tool
def search_private_knowledge(query: str) -> str:
    """当需要查询用户的个人背景、私有项目（如 SHIFT）或特定事实时，使用此工具进行检索。"""
    relevant_docs = retriever.invoke(query)
    return "\n".join(doc.page_content for doc in relevant_docs)


tools = [search_private_knowledge]

# 3. DeepSeek 模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
    temperature=0,
)

agent_executor = create_react_agent(llm, tools)

# 4. 运行
query = "William 会什么编程语言，他的项目叫什么？"
print(f"用户提问: {query}\n")

response = agent_executor.invoke({"messages": [("user", query)]})
print("Agent 回复:")
print(response["messages"][-1].content)
