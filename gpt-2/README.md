# Learning source

* Lets reproduce GPT-2 https://www.youtube.com/watch?v=l8pRSuU81PU

## GPT-2

GPT-2 is a decoder only transformer

GPT-2 has no encoder

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

## Note:
INT8 is used for inference, but not training. INT8 has a uniform spacing. we shall use float to have a better match of the normal distribution of weights during the training of neural networks.

For training bfloat16 has more TFLOPS than the standard float32.

GPU Memory Bandwidth: the speed of which you can access the GPU memory.






