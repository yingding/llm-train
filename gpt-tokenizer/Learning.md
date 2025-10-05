# Learning source

* Let's build the GPT Tokenizer https://www.youtube.com/watch?v=zduSFxRajkE

## Tokenization

Tokenization is at the heart of much weirdness of LLMs. Do not brush it off.

- Why can't LLM spell words? **Tokenization**.
- Why can't LLM do super simple string processing tasks like reversing a string? **Tokenization**.
- Why is LLM worse at non-English languages (e.g. Japanese)? **Tokenization**.
- Why is LLM bad at simple arithmetic? **Tokenization**.
- Why did GPT-2 have more than necessary trouble coding in Python? **Tokenization**.
- Why did my LLM abruptly halt when it sees the string "<|endoftext|>"? **Tokenization**.
- What is this weird warning I get about a "trailing whitespace"? **Tokenization**.
- Why the LLM break if I ask it about "SolidGoldMagikarp"? **Tokenization**.
- Why should I prefer to use YAML over JSON with LLMs? **Tokenization**.
- Why is LLM not actually end-to-end language modeling? **Tokenization**.
- What is the real root of suffering? **Tokenization**.


## Progress
Tokenization Web Demo App
https://youtu.be/zduSFxRajkE?t=368
(6:08)

Unicode - python ord()
https://youtu.be/zduSFxRajkE?t=1016
(16:56)

BPM - merge the most frequent token pair with new token, utf-8 ranges `[0, 255]` replace the first token pair with `256`
https://youtu.be/zduSFxRajkE?t=1916
(31:56)

**GOON**

## Notes

For the same sentence from english translated to german (non-english), the tokenizer will break up the sentence to more tokens, als the english sentence. For the same meaning, you have less context window, since the tokens are more used for non-english as english.

| Language | Tokenizer | Sentence |  Number of Words | tokens | Number of Tokens | tokens-words ratio |
|----------|-------|----------|------------------|--------|------------------|---------------------|
| English  | GPT-2 | Hello World, how are you doing | 6 words | 15496, 2159, 11, 703, 389, 345, 1804 | 7 tokens | 7/6 = 1.17 |
| German   | GPT-2 | Hallo Welt, wie geht es dir   | 6 words | 34194, 78, 370, 2120, 11, 266, 494, 4903, 4352, 1658, 26672, 220 | 12 tokens | 12/6= 2.0 |
| English  | GPT-4 | Hello World, what a wonderful day| 6 words | 9906, 4435, 11, 1148, 264, 11364, 1938 | 7 tokens | 7/6= 1.17 |
| German   | GPT-4 | Hallo Welt, was für ein wundervoller Tag   | 7 words | 79178, 46066, 11, 574, 7328, 4466, 289, 1263, 651, 70496, 12633 | 11 tokens | 11/7= 1.57 |

Tokenizer has a smaller "tokens-words ratio" for english language, since the training dataset for LLM has more english text than german text, thus the tokenizer can have a large chunk for english words.

GPT-4 tokenizer (cl100k_base)

## Encoding
Unicode standard is constantly changing, which is not stable.
https://www.reedbeta.com/blog/programmers-intro-to-unicode/

The "utf-8" encoding has a value range of 256, with this encoding chunks will be small and token sequence for the prompt will be very long. With a finite context windows, long token sequence for same input text is inefficient.

BPE will compress the "utf-8" raw byte encoding so that the compressed encoding can be passed to the transformer.

## Tokenization-free autoreressive sequence modeling
Using the raw "utf-8" byte string without tokenizer
Modified the transformer to be hierachical transformer to feed in raw utf-8 byte

MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers: https://arxiv.org/pdf/2305.07185

## BPE
Byte Pair Encoding: the input sequence is too long, we would like to compress it.
1. Find the pair of token, that ocurs the most frequently.
2. We replace the pair of tokens with a single new token, and append it to vocabulary (lookup table for token to char)
3. repeat whole proccess for new most frequently pair of tokens (to iteratively compress the sequence)
