# LangChain RAG Agent

一个极简的 RAG Agent 示例：使用 LangGraph 的 ReAct Agent，通过检索工具查询私有知识库，由 DeepSeek 模型生成回答。

## 技术栈

- **LLM**：DeepSeek（`deepseek-chat`，OpenAI 兼容接口）
- **Agent**：LangGraph `create_react_agent`
- **向量库**：FAISS
- **Embedding**：HuggingFace `BAAI/bge-small-zh-v1.5`（本地运行）

## 快速开始

### 1. 安装依赖

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install python-dotenv langchain-openai langchain-huggingface \
            langchain-community langgraph sentence-transformers faiss-cpu
```

### 2. 配置密钥

在项目根目录创建 `.env` 文件：

```
DEEPSEEK_API_KEY=你的密钥
```

> `.env` 已被 `.gitignore` 忽略，请勿提交到仓库。

### 3. 运行

```bash
python agent.py
```

首次运行会下载 embedding 模型，可能需要一些时间。

## 工作原理

1. 将知识库文档向量化，存入 FAISS
2. 把检索封装成 Tool（`search_private_knowledge`）
3. Agent 根据问题决定是否调用检索工具
4. 结合检索结果由 DeepSeek 生成最终回答

## 自定义知识库

修改 `agent.py` 中的 `docs` 列表即可：

```python
docs = [
    Document(page_content="你的内容……"),
]
```

## 注意

- 使用 DeepSeek API 需要账户有可用余额，否则会返回 `402 Insufficient Balance`
- 不要把 API 密钥写进源码或提交到 Git
