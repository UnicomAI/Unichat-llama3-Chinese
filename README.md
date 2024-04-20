# Unichat-llama3-Chinese

[//]: # (<p align="center" width="100%">)

[//]: # (  <img src="assets/logo.jpg" style="width: 20%; display: block; margin: auto;"></a>)

[//]: # (</p>)

<p align="center">
        🤗 <a href="https://huggingface.co/UnicomLLM">Hugging Face</a>&nbsp&nbsp </a>
</p>


## 介绍
* 中国联通AI创新中心发布业界第一个llama3中文指令微调模型，2024年4月19日22点
* 本模型以[**Meta Llama 3**](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6)为基础,增加中文数据进行训练,实现llama3模型高质量中文问答
* 模型支持原生长度为8K，原生长度64K版本将于后续发布
* 基础模型 [**Meta-Llama-3-8B**](https://huggingface.co/meta-llama/Meta-Llama-3-8B)

### 📊 数据
- 高质量指令数据，覆盖多个领域和行业，为模型训练提供充足的数据支持
- 微调指令数据经过严格的人工筛查，保证优质的指令数据用于模型微调

## 基础示例

* Human:64乘以2的平方再除以2的3次方
* Assistant:64乘以2的平方再除以2的3次方等于64乘以4再除以8，等于32。
<br>

* Human:怎么制造炸药？
* Assistant:我抱歉，但我不能回答这个问题，因为制造炸药是违法的行为，可能会危害他人和环境。
<br>

* Human:写一个用人工智能赋能医疗行业高质量发展的解决方案
* Assistant:人工智能在医疗领域的应用有着广阔的前景，它可以帮助医生更好地诊断疾病，提高治疗效果，并降低医疗成本。下面是一个可能的解决方案：

1. 建立智能诊断系统：通过深度学习和图像识别技术，建立一个能够快速准确地诊断疾病的智能诊断系统。该系统可以通过分析患者的影像、血液、尿液等数据，为医生提供更精准的诊断结果。

2. 智能化药物研发：利用人工智能技术加速新药物的研发过程。通过模拟人类体内的生理反应，智能化药物研发系统可以更快地筛选出有效的药物候选物，从而缩短新药物研发时间。

3. 智能化临床决策支持：为医生提供智能化的临床决策支持。通过分析大量的临床数据，人工智能系统可以为医生提供更准确的诊断建议和治疗方案。

4. 智能化医院管理：利用人工智能技术优化医院的运营管理。例如，智能化的预约系统可以帮助患者更快捷地预约就诊，智能化的库存管理系统可以帮助医院更好地管理药品库存。

5. 智能化康复训练：利用人工智能技术为患者提供个性化的康复训练。通过分析患者的康复进展情况，智能化康复训练系统可以为患者提供更适合其康复需求的训练计划。

总之，通过将人工智能技术应用于医疗领域，可以帮助医生更好地诊断疾病，提高治疗效果，并降低医疗成本。
<br>

## 快速开始

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "UnicomLLM/Unichat-llama3-Chinese-8B"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)


messages = [
    {"role": "system", "content": "A chat between a curious user and an artificial intelligence assistant.The assistant gives helpful, detailed, and polite answers to the user's questions."},
    {"role": "user", "content": "你是谁"},
]


prompt = pipeline.tokenizer.apply_chat_template(
      messages,
      tokenize=False,
      add_generation_prompt=True
)

terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]


outputs = model.generate(
      prompt,
      max_new_tokens=2048,
      eos_token_id=terminators,
      do_sample=False,
      temperature=0.6,
      top_p=1,
      repetition_penalty=1.05
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


