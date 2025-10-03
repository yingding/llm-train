# Learning source

* Lets reproduce GPT-2 https://www.youtube.com/watch?v=l8pRSuU81PU

## GPT-2

GPT-2 is a decoder only transformer
GPT-2 has no encoder

## About this repo
This repo documents the pretraining steps for build a GPT2 124M model from scratch.
The result model is only capable of predict the next token.

If you want to talk to the model, you will need to fine-tune the pretrained model to a chat format using supervised fine-tuning (SFT) by using further training datasets with a "user", "assistant" structure. Swap out the datasets and continue the training.

Wrap it up: **pretraining** phase of the LLM is documented in this repo, without *fine tuning* phase.

## Learning sections
* GPT-2 model code generation test (41:23) https://youtu.be/l8pRSuU81PU?t=2483
* GPT-2 architecture workthrough (13:56) https://youtu.be/l8pRSuU81PU?t=836
* Attension implementation (24:00) https://youtu.be/l8pRSuU81PU?t=1440

## online openai tiktokenizer
* https://tiktokenizer.vercel.app

## monitor on apple silicon acceletor
```shell
sudo asitop
```
* asitop https://github.com/tlkh/asitop

## Progress
Eleuther Eval Harness (Eval dataset)
https://youtu.be/l8pRSuU81PU?t=14102
(3:55:02)

Shuffle documents order in the training set
https://youtu.be/l8pRSuU81PU?t=13791
(3:49:52)

HellaSwag Evaluation Dataset
https://youtu.be/l8pRSuU81PU?t=12526
(3:28:46)

Validation, Logging
https://youtu.be/l8pRSuU81PU?t=12192
(3:23:12)

Open WebText, Common Crawl (filtered), SlimPajama (subset of RedPajam), FineWeb, FineWeb-Edu dataset
https://youtu.be/l8pRSuU81PU?t=11423
(3:10:23)

Distributed Data Parallel
https://youtu.be/l8pRSuU81PU?t=10015
(2:46:55)
**Goon**

Gradient Accumulation
https://youtu.be/l8pRSuU81PU?t=9403
(2:36:43)

FlashAttention
https://youtu.be/l8pRSuU81PU?t=7235
(2:00:35)

torch.compile
https://youtu.be/l8pRSuU81PU?t=6508
(1:48:29)

BF16 training
https://youtu.be/l8pRSuU81PU?t=5382
(1:29:42)

Training precision, dtype and TFLOPS of GPU
https://youtu.be/l8pRSuU81PU?t=5020
(1:23:40)

https://youtu.be/l8pRSuU81PU?t=3178
(52:58)

https://youtu.be/l8pRSuU81PU?t=2619
(43:39)

## References:
* GPT-2 paper https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
* GPT-3 paper https://arxiv.org/abs/2005.14165
* Attention Is All You Need (original transformer paper) https://arxiv.org/abs/1706.03762
* PyTorch GELU https://pytorch.org/docs/stable/generated/torch.nn.GELU.html
* GELU paper https://arxiv.org/abs/1606.08415
* BF16 in pytorch (Automatic Mixed Precision) https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
* torch compile https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
* Flash Attention paper: https://arxiv.org/pdf/2205.14135

## LLM Datasets, Data Mixture
### 1. FineWeb (Common Crawl filtered)
* Paper https://arxiv.org/abs/2406.17557
* Huggingface FineWeb (sample-10BT) https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
* Huggingface FineWeb-edu (sample-10BT) https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

### 2. HellaSwag eval
Sentence completion dataset, provides early signal of improvement for small sized language model
* Paper https://arxiv.org/pdf/1905.07830
* Dataset https://rowanzellers.com/hellaswag/

### 3. Eleuther Eval Harness eval
Further model evaluation dataset and methods
* repo: https://github.com/EleutherAI/lm-evaluation-harness


## cosine decay learning rate curve
* cosine decay learning rate schedule with warmup https://miro.medium.com/v2/resize:fit:720/format:webp/1*BJCssPOCn4u__NoAZs392w.png
* cosine leanring rate decay https://scorrea92.medium.com/cosine-learning-rate-decay-e8b50aa455b

## learning with batches
You most learning to ignore the tokens that don't come up in your training set very often.
You are learning very simple biases. Every simple you put through your network, is basically telling whether to use these tokens or not.
The gradients from every single example are actually extremely highly correlated.
They all look roughly the same.
Later in the optimization, once you have learned all the simple stuff. 
That is where the actual work starts. That is where the gradients become more decorrelated.

## Note:
INT8 is used for inference, but not training. INT8 has a uniform spacing. we shall use float to have a better match of the normal distribution of weights during the training of neural networks.

For training bfloat16 has more TFLOPS than the standard float32.

GPU Memory Bandwidth: the speed of which you can access the GPU memory.