from openai import OpenAI
import os

os.environ["http_proxy"]="127.0.0.1:7890"
os.environ["https_proxy"]="127.0.0.1:7890"

client = OpenAI(
    api_key = "sk-vjLEzi6oQ6SYyqtFymRhHGWTApKQJNOezPujSgVjtLokkBTu",
    base_url = "https://api.agicto.cn/v1"
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "讲个笑话",
        }
    ],
    model="gpt-4o-mini", #此处更换其它模型,请参考模型列表 eg: google/gemma-7b-it
)

print(chat_completion.choices[0].message.content)