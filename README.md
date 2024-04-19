# Unichat-llama3-Chinese

[//]: # (<p align="center" width="100%">)

[//]: # (  <img src="assets/logo.jpg" style="width: 20%; display: block; margin: auto;"></a>)

[//]: # (</p>)

<p align="center">
        🤗 <a href="https://huggingface.co/UnicomLLM">Hugging Face</a>&nbsp&nbsp </a>
</p>

## 介绍
* 中国联通发布业界第一个llama3中文模型
* 本模型以[**Meta Llama 3**](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6)为基础,增加中文数据进行训练,实现llama3模型高质量中文问答


### 📊 数据
- 中国联通自有数据，覆盖多个领域和行业，为模型训练提供充足的数据支持。
- 基于中国联通统一数据中台，归集公司内外部等多种类型数据，构建中国联通高质量数据集


### 最新动态
2024年04月19日： 第一版模型发布 [**Unichat-llama3-Chinese**](https://huggingface.co/UnicomLLM)

## 快速开始

### 🤗 Hugging Face Transformers
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_id = "UnicomLLM/Unichat-llama3-Chinese-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Who are you?"},
]
input_ids = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt"
).to(model.device)
terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
outputs = model.generate(
    input_ids,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))
```


## 模型下载

### Llama3中文模型
| 模型名称                     | 🤗模型加载名称             | 下载地址                                                     |
|--------------------------| ------------------------- | --------------------- |
| Unichat-llama3-Chinese-8B | UnicomLLM/Unichat-llama3-Chinese-8B  | [HuggingFace](https://huggingface.co/UnicomLLM/Unichat-llama3-Chinese-8B)  |


### Llama3官方模型

| 模型名称   | 🤗模型加载名称             | 下载地址                                                     |
| ---------- | ------------------------- | --------------------- |
| Llama3-8B  | meta-llama/Meta-Llama-3-8B  | [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B)  |
| Llama3-8B-Chat  | meta-llama/Meta-Llama-3-8B-Instruct  | [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)  |
| Llama3-70B | meta-llama/Meta-Llama-3-70B | [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-7B)  |
| Llama3-70B-Chat  | meta-llama/Meta-Llama-3-70B-Instruct  | [HuggingFace](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct)  |



## Web UI

### Text generation web UI
可使用 [text-generation-webui](https://github.com/oobabooga/text-generation-webui)  框架部署网页demo.

## 模型部署
可使用以下框架,实现模型本地部署
- [vllm](https://github.com/vllm-project/vllm) 
- [sglang](https://github.com/sgl-project/sglang) 
- [text-generation-inference](https://github.com/huggingface/text-generation-inference)


## 模型微调
可使用以下框架, 对模型进行SFT, LoRA, DPO, PPO等方式的微调
- [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Llama-Factory](https://github.com/hiyouga/LLaMA-Factory)


