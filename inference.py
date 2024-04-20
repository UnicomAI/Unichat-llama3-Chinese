import transformers
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
    {"role": " assistant", "content": "您好，我是一个人工智能助手，一个由人工开发的 AI大模型，我可以回答各种问题并提供必要的支持。"},
    {"role": "user", "content": "写一个用人工智能赋能医疗行业高质量发展的解决方案"}
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

outputs = pipeline(
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
