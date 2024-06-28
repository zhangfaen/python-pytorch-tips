import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# 初始化模型和分词器
model_name = "openlm-research/open_llama_3b"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# 编码输入文本
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")

# 创建目标标签
labels = inputs["input_ids"].clone()

import pdb
pdb.set_trace()

# 假设我们想忽略第二个token的损失计算
IGNORE_INDEX = -100
labels[0, 1] = IGNORE_INDEX

# 模型前向传播
outputs = model(input_ids=inputs["input_ids"], labels=labels)

# 输出损失
print(outputs.loss)
