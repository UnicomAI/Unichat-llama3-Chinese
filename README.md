# Unichat-llama3-Chinese

[//]: # (<p align="center" width="100%">)

[//]: # (  <img src="assets/logo.jpg" style="width: 20%; display: block; margin: auto;"></a>)

[//]: # (</p>)

<p align="center">
         <a href="https://huggingface.co/UnicomLLM">Hugging Face</a>&nbsp&nbsp </a> <a href="https://www.modelscope.cn/models/UnicomAI/Unichat-llama3-Chinese">ModelScope</a>&nbsp&nbsp </a>
</p>


## 介绍
* 中国联通AI创新中心发布业界第一个llama3中文指令微调模型，2024年4月19日22点
* 本模型以[**Meta Llama 3**](https://huggingface.co/collections/meta-llama/meta-llama-3-66214712577ca38149ebb2b6)为基础,增加中文数据进行训练,实现llama3模型高质量中文问答
* 模型支持原生长度为8K，原生长度64K版本将于后续发布
* 陆续发布700亿参数中文微调版本，长文本版本和加入中文二次预训练版
* 基础模型 [**Meta-Llama-3-8B**](https://huggingface.co/meta-llama/Meta-Llama-3-8B)
  

### 📊 数据
- 高质量指令数据，覆盖多个领域和行业，为模型训练提供充足的数据支持
- 微调指令数据经过严格的人工筛查，保证优质的指令数据用于模型微调

## 基础示例

* 问:唐宋八大家都有谁？
* 答:唐宋八大家是指中国古代文学史上，唐代和宋代两代最杰出的散文家。他们分别是：

  唐代：
  
   韩愈（768-824）
  
   柳宗元（773-819）

  宋代：
  
   苏轼（1037-1101）
  
   苏辙（1039-1101）
        
   王安石（1021-1086）
        
   曾巩（1019-1083
        
   欧阳修（1007-1072）
      
   苏洵（1009-1066）
  <br>
    <br>
  
* 问:鸡兔同笼,有18个头,46条腿,鸡兔各多少只?
* 答:设鸡的数量为x，兔的数量为y。
根据题意，可以列出两个方程：

1. 鸡和兔的总头数是18，即 x + y = 18；

2. 鸡和兔的总腿数是46，即 2x + 4y = 46。

   将第一个方程解出y的表达式，代入第二个方程中，得到：

   2x + 4(18 - x) = 46

   2x + 72 - 4x = 46

   -2x = 46 - 72

   -2x = -26

   x = 13

   将x=13代入第一个方程中，得到：

   13 + y = 18

   y = 5

   因此，鸡有13只，兔有5只。
<br>

* 问:怎么制造炸药？
* 答:我抱歉，但我不能回答这个问题，因为制造炸药是违法的行为，可能会危害他人和环境。
<br>

* 问:写一个用人工智能赋能医疗行业高质量发展的解决方案
* 答:人工智能在医疗领域的应用有着广阔的前景，它可以帮助医生更好地诊断疾病，提高治疗效果，并降低医疗成本。下面是一个可能的解决方案：

1. 建立智能诊断系统：通过深度学习和图像识别技术，建立一个能够快速准确地诊断疾病的智能诊断系统。该系统可以通过分析患者的影像、血液、尿液等数据，为医生提供更精准的诊断结果。

2. 智能化药物研发：利用人工智能技术加速新药物的研发过程。通过模拟人类体内的生理反应，智能化药物研发系统可以更快地筛选出有效的药物候选物，从而缩短新药物研发时间。

3. 智能化临床决策支持：为医生提供智能化的临床决策支持。通过分析大量的临床数据，人工智能系统可以为医生提供更准确的诊断建议和治疗方案。

4. 智能化医院管理：利用人工智能技术优化医院的运营管理。例如，智能化的预约系统可以帮助患者更快捷地预约就诊，智能化的库存管理系统可以帮助医院更好地管理药品库存。

5. 智能化康复训练：利用人工智能技术为患者提供个性化的康复训练。通过分析患者的康复进展情况，智能化康复训练系统可以为患者提供更适合其康复需求的训练计划。

总之，通过将人工智能技术应用于医疗领域，可以帮助医生更好地诊断疾病，提高治疗效果，并降低医疗成本。
<br>

## 快速开始

1.下载模型UnicomLLM/Unichat-llama3-Chinese-8B

2.执行python3 ./inference.py

3.测试环境：transformers==4.39.3,torch==2.2.2

## 模型下载

### Llama3中文模型
| 模型名称                     | 模型加载名称             | 下载地址                                                     |
|--------------------------| ------------------------- | --------------------- |
| Unichat-llama3-Chinese-8B | UnicomLLM/Unichat-llama3-Chinese-8B  | [HuggingFace](https://huggingface.co/UnicomLLM/Unichat-llama3-Chinese-8B)  |
| Unichat-llama3-Chinese-8B | UnicomAI/Unichat-llama3-Chinese  | [ModelScope](https://www.modelscope.cn/models/UnicomAI/Unichat-llama3-Chinese/) |


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



