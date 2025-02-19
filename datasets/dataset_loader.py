from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

import os
os.environ["http_proxy"] = "127.0.0.1:7890"
os.environ["https_proxy"] = "127.0.0.1:7890"

tokenizer = AutoTokenizer.from_pretrained("kwaikeg/kagentlms_qwen_7b_mat", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    "kwaikeg/kagentlms_qwen_7b_mat",
    device_map="auto",
    trust_remote_code=True
).eval()

response, history = model.chat(tokenizer, "你好", history=None)
print(response)