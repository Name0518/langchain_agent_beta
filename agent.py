from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os
load_dotenv()

model = init_chat_model(
    model="deepseek-v4-pro",
    model_provider="deepseek",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    temperature=0.7,
)

response = model.invoke("你好，简单介绍下自己。")
print(response.content)
