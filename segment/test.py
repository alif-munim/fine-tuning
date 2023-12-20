import os
import torch
import torch.nn as nn

from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModel,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
)
from peft import LoraConfig, PeftModel, LoraModel, get_peft_model
from trl import SFTTrainer

import os
from datasets import load_dataset, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np

model_name = "meta-llama/Llama-2-7b-hf"

lora_r = 64
lora_alpha = 16
lora_dropout = 0.1

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

report_to = "wandb"
output_dir = "./results"
num_train_epochs = 1
fp16 = True
bf16 = False
per_device_train_batch_size = 1
per_device_eval_batch_size = 1
gradient_accumulation_steps = 16
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
# weight_decay = 0.001
weight_decay = 0.0

# Use these if the dataset has a test or validation split
# ValueError: Trainer: evaluation requires an eval_dataset.
evaluation_strategy = "steps"
eval_steps = 187

optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
warmup_ratio = 0.03
group_by_length = True

save_steps = 0
logging_steps = 10
max_seq_length = None
packing = False
# device_map = {"": 0}

dataset_path = "/scratch/alif/timdettmers___json/timdettmers--openassistant-guanaco-c93588435bc90172/0.0.0/fe5dd6ea2639a6df622901539cb550cf8797e5a6b2dd7af1cf934bed8e233e6e/"
train_path = os.path.join(dataset_path, 'json-train.arrow')
test_path = os.path.join(dataset_path, 'json-test.arrow')

train_dataset = Dataset.from_file(train_path)
test_dataset = Dataset.from_file(test_path)

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)
        fp16 = False
        bf16 = True
    
    
# Load base model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    # device_map=device_map
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# By default, LoRA is only applied to attention layers
# https://github.com/huggingface/peft/blob/main/src/peft/utils/constants.py#L49
# The QLoRA paper suggests that LoRA should be applied to all linear layers
# https://github.com/huggingface/peft/issues/735
target_modules = [name for name, layer in model.named_modules() if isinstance(layer, nn.Linear)]

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules # Applying LoRA to all linear layers
)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

# lora_model = LoraModel(peft_config, model, "default")

lora_model = get_peft_model(model, peft_config)
print_trainable_parameters(model)
print_trainable_parameters(lora_model)








# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# from transformers import (
#     AutoModelForCausalLM,
#     AutoModel,
#     AutoTokenizer
# )

# print("Initializing GTE embedding model...")

# class GTEEmbeddingModel(nn.Module):
#     def __init__(self):
#         super(GTEEmbeddingModel, self).__init__()
#         self.tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
#         self.model = AutoModel.from_pretrained("thenlper/gte-base")

#     def forward(self, input_texts):
#         batch_dict = self.tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
#         outputs = self.model(**batch_dict)
#         embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
#         return F.normalize(embeddings, p=2, dim=1)

#     def average_pool(self, last_hidden_states, attention_mask):
#         last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
#         return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    
# print("Successfully initialized embedding model!")

# def cluster_dataset(dataset, num_clusters, embedding_model):    
#     # Extract text data
#     texts = [entry['text'] for entry in dataset]

#     # Generate embeddings
#     embeddings = embedding_model(texts)

#     # Clustering
#     kmeans = KMeans(n_clusters=num_clusters, random_state=0)
#     kmeans.fit(embeddings.detach().numpy())

#     # Assigning cluster labels to each text
#     cluster_labels = kmeans.labels_

#     # Creating a DataFrame for easier visualization
#     clustered_data = pd.DataFrame({'text': texts, 'cluster': cluster_labels})

#     return clustered_data


# embedding_model = GTEEmbeddingModel()


# import torch.nn.functional as F
# from torch import Tensor
# from transformers import AutoTokenizer, AutoModel

# def average_pool(last_hidden_states: Tensor,
#                  attention_mask: Tensor) -> Tensor:
#     last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
#     return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# input_texts = [
#     "what is the capital of China?",
#     "how to implement quick sort in python?",
#     "Beijing",
#     "sorting algorithms"
# ]

# tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
# model = AutoModel.from_pretrained("thenlper/gte-base")

# # Tokenize the input texts
# batch_dict = tokenizer(input_texts, max_length=512, padding=True, truncation=True, return_tensors='pt')

# outputs = model(**batch_dict)
# embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# # (Optionally) normalize embeddings
# embeddings = F.normalize(embeddings, p=2, dim=1)
# scores = (embeddings[:1] @ embeddings[1:].T) * 100
# print(scores.tolist())


