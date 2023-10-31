## Understanding Language Models
Since 2021, there have been enormous advances in the field of NLP, most notably GPT-4 and ChatGPT. The aim of this repository is to understand the mechanics behind powerful language models by coding and training them from scratch. Currently, I am working on understanding the LLaMA model and using LoRA/qLoRA to train it more efficiently.


### GPT
> Jump to the [karpathy-gpt](https://github.com/alif-munim/language-models/tree/main/karpathy-gpt) directory. <br/>

Starting from a simple bigram language model, building out GPT from scratch. The completed GPT model be trained on character-level text data (e.g. tiny shakespeare) to generate convincing shakespeare-like text. Key concepts include:
1. Self-Attention
2. Scaled Dot-Product Attention
3. Multi-Head Self-Attention
4. Layer Normalization

### LLaMA
> Jump to the [llama](https://github.com/alif-munim/language-models/tree/main/llama) directory. <br/>

Building out the LLaMA 2 language model from Meta AI. The model can load pre-trained weights and perform inference. Key concepts include:
1. Rotary Positional Embeddings
2. KV-Cache
3. Grouped-Query Attention
4. RMSNorm
5. SwiGLU Activation Function

### LoRA
> Jump to the [lora](https://github.com/alif-munim/language-models/tree/main/lora) directory. <br/>

LoRA is a fine-tuning method that drastically reduces the number of trainable parameters in pre-trained language models by adding two trainable weight matrices to the original model weights. It is based on singular value decomposition, in which a large matrix is broken down into its component eigenvalues and eigenvectors. The most important information in a matrix is often contained in just the first few singular values. Key concepts include:
1. Singular Value Decomposition

### Min-LoRA
> Jump to the [lora](https://github.com/alif-munim/language-models/tree/main/min-lora) directory. <br/>

A minimal reproduction of adding LoRA to fine-tune a pre-trained GPT2 model from OpenAI. A starting point for my own LLM fine-tuning method, adapted from diffusion models.
