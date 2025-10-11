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

What mean Training of Tokenizer?: Using BPM on the Tokenizer training dataset to determine the token vocabulary
https://youtu.be/zduSFxRajkE?t=2360
(39:20)

Decoding of tokenizer
https://youtu.be/zduSFxRajkE?t=2570
(42:50)

Encoding of tokenizer
https://youtu.be/zduSFxRajkE?t=2926
(48:46)

Tokenizer in GPT Series (GPT2)
https://youtu.be/zduSFxRajkE?t=3457
(57:37)

Regex pattern for enforcing BPM exception
https://youtu.be/zduSFxRajkE?t=3636
(1:00:36)

Tiktoken lib for tokenization by openai
https://youtu.be/zduSFxRajkE?t=4304
(1:11:44)

Walk through gpt-2 `encoder.py` file
https://youtu.be/zduSFxRajkE?t=4506
(1:15:07)

Encoder, Byteencoder in openai `encoder.py`
https://youtu.be/zduSFxRajkE?t=4603
(1:16:43)

Special tokens
https://youtu.be/zduSFxRajkE?t=4709
(1:18:29)

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

https://en.wikipedia.org/wiki/Byte-pair_encoding

## Tokenizer Training
The Tokenizer is a completely separate, **independent** module from the LLM. 

Tokenizer has its own training dataset of text (which could be different from that of the LLM), on which you train the vocabulary using the Byte Pair Encoding (BPE) algorithm. It then translates back and forth between raw text and sequences of tokens. The LLM later only ever sees the tokens and never directly deals with any text.

The training of BPM, in the sense of most frequent token pairs seen in the training set of the tokenizer. The training set determines how the text will be compressed to token vocabulary.

Tokenizer is a translation layer between raw text and token sequence, by using the encoding and decoding step.

You may want to have the training set be different between the tokenizer and LLM. While training the tokenizer, we care about different languages, code or not-code. More data to a language during the tokenizer training will allow more merges to be done by the tokenizer, so that the LLM will have more densed token to work with, since the context window is finite for LLM.

(Better training of tokenizer, will be to have more merges to allow more information to be condense into token-space )

## Tokenizer Decoding
Given a sequence of integers in the range [0, vocab_size], what is the text?

While decoding the bytes, multi-bytes need to following start bytes, use `tokens.decode("utf-8", errors="replace")` to replace the start byte
https://en.wikipedia.org/wiki/UTF-8

## BPE enforced token merging rules
From the Radford2019, the many variants of token with punctuations e.g. `dog.`, `dog!`, `dog?` will be merged as new tokens in the vocabulary. This results in a sub-optimal allocation of limited vocabulary slots and model capacity. To avoid this, rules are added to prevent BPE from merging across character categories for any byte sequence.

GPT-2 `encoder.py` is the encode and decode: https://github.com/openai/gpt-2/blob/master/src/encoder.py

`regex` is an extension of `re` in python, a more powerful `re` version.

GPT-2 `encoder.py` uses the regex pattern `gpt2pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")` to split input text into token list first.
All the element of this splitted list can be processed independently by the tokenizer, so that only the tokens inside each of splitted list can be merged by the BPE. By splitting up text in this way, gpt-2 tokenizer will not merge the token cross works and punctuations.

The `encoder.py` is only the inference code, the training code for tokenizer for gpt-2 is not released by openai. By the web gpt-2 tokenizer, the spaces in python code are not merged by the tokenizer in the vocabulary. There must be addiontal rules applied by the training code of gpt-2 by openai.

## Tiktoken Fast BPE inference lib
In the tiktoken code https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py

```python
# The pattern in the original GPT-2 release is:
# r"""'s|'t|'re|'ve|'m|'ll|'d| ?[\p{L}]+| ?[\p{N}]+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
# This is equivalent, but executes faster:
r50k_pat_str = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}++| ?\p{N}++| ?[^\s\p{L}\p{N}]++|\s++$|\s+(?!\S)|\s"""
)
```
openai changes the pattern for split up the text.

`cl100K_base` gpt-4 tokenizer has different text split pattern. All the space charators are grouped together. 
`\p{N}{1,3}+` only split up to 3 digits of number, only 3 number of digits will be merged as token in the BPE added to the new token vocabulary in gpt4.
```python
"pat_str": r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
```

In the [encoder.py](https://github.com/openai/gpt-2/blob/master/src/encoder.py) file, openai is saving the `encoder.json` (base vocab), `vocab.bpe` (bpe_merges for merged new vocab from BPE training).

With the `base vocab` and `bpe_merges` files, you can save the tokenizer for encoding and decoding.









