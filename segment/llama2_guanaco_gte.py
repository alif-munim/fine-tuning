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
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

import os
from datasets import load_dataset, Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from itertools import islice

def preprocess_instruct(examples):
    # Concatenate 'prompt' and 'completion' fields
    texts = [prompt + " " + completion for prompt, completion in zip(examples['prompt'], examples['completion'])]
    return {'text': texts}

num_clusters = 4  # Adjust the number of clusters as needed
resume_from = 0
model_name = "meta-llama/Llama-2-7b-hf"
dataset = "instruct"
cluster = "narval"
attention_only = True
layer_config = "att" if attention_only else "lin"

if cluster == "cedar":
    if dataset == "guanaco":
        dataset_name = "timdettmers/openassistant-guanaco"
    elif dataset == "instruct":
        dataset_name = "monology/VMware-open-instruct-higgsfield"
    train_dataset = load_dataset(dataset_name, split="train")
    print(f"Training dataset set to {dataset_name} from hugging face")

if cluster == "narval":
    if dataset == "guanaco":
        dataset_path = "/scratch/alif/timdettmers___json/timdettmers--openassistant-guanaco-c93588435bc90172/0.0.0/fe5dd6ea2639a6df622901539cb550cf8797e5a6b2dd7af1cf934bed8e233e6e/json-train.arrow"
    elif dataset == "instruct":
        dataset_path = '/scratch/alif/monology___v_mware-open-instruct-higgsfield/default/0.0.0/622a7cf65a222fcb/v_mware-open-instruct-higgsfield-train.arrow'
    train_dataset = Dataset.from_file(dataset_path)
    train_dataset = train_dataset.map(preprocess_instruct, batched=True)
    print(f"Training dataset set to: {dataset} from local path: {dataset_path}")

lora_r = 64
lora_alpha = 16
lora_dropout_factor = 0
lora_dropout = lora_dropout_factor * 0.1

new_model = f"llama-2-7b-{dataset}-lora-{layer_config}-d{lora_dropout_factor}-r{lora_r}-a{lora_alpha}"

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

report_to = "wandb"
output_dir = new_model
num_train_epochs = 1
fp16 = True
bf16 = False
per_device_train_batch_size = 8
gradient_accumulation_steps = 2
gradient_checkpointing = True
max_grad_norm = 0.3
learning_rate = 2e-4
# weight_decay = 0.001
weight_decay = 0.0
max_steps = 1875

# Use these if the dataset has a test or validation split
# ValueError: Trainer: evaluation requires an eval_dataset.
# evaluation_strategy = "steps"
# eval_steps = 187
# per_device_eval_batch_size = 1

optim = "paged_adamw_32bit"
lr_scheduler_type = "cosine"
warmup_ratio = 0.03
group_by_length = True

save_steps = 500
logging_steps = 10
max_seq_length = None
packing = False
# device_map = {"": 0}

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

if attention_only:
    print(f'Configuring LoRA to be applied to attention layers...')
    # Load LoRA configuration
    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
else:
    print(f'Configuring LoRA to be applied to all transformer linear layers...')
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

#### CLUSTERING #########

print("Loading embeddings...")

if dataset == "instruct":
    embeddings = np.load('instruct_embeddings.npy')
elif dataset == "guanaco":
    embeddings = np.load('embeddings.npy')

def cluster_dataset(dataset, num_clusters, embeddings):    
    # Extract text data
    texts = [entry['text'] for entry in dataset]

    # Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(embeddings)

    # Assigning cluster labels to each text
    cluster_labels = kmeans.labels_

    # Creating a DataFrame for easier visualization
    clustered_data = pd.DataFrame({'text': texts, 'cluster': cluster_labels})

    return clustered_data


# Cluster the dataset using precomputed embeddings
print(f'Creating {num_clusters} clusters from precomputed embeddings...')
clustered_data = cluster_dataset(train_dataset, num_clusters, embeddings)
unique_clusters = clustered_data['cluster'].unique()
cluster_datasets = {}

# Loop through each cluster and create datasets
for cluster_label in unique_clusters:
    cluster_df = clustered_data[clustered_data['cluster'] == cluster_label]
    cluster_datasets[f"cluster_{cluster_label}"] = Dataset.from_pandas(cluster_df)
    

count = 1
total = len(cluster_datasets.items())

for cluster_label, cluster_dataset in islice(cluster_datasets.items(), resume_from, total, 1):
    
    
    # The QLoRA paper uses a batch size of 16 for a total of 1875 steps for Guanaco (~10K)
    # Which means the model sees 30K examples, or each example 3 times
    # However, Sebastian Raschka's experiments suggest multiple iterations over a dataset harms performance
    dataset_iterations = 1
    cluster_size = len(cluster_dataset)
    cluster_max_steps = dataset_iterations * (cluster_size // per_device_train_batch_size)
    print(f'{count}/{total} Training {cluster_label} for {cluster_max_steps} steps...')
    
    new_model_name = f"llama-2-7b-{dataset}_lora-{layer_config}-d{lora_dropout_factor}-r{lora_r}-a{lora_alpha}-{num_clusters}_{cluster_label}"
    
    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=new_model_name,
        max_steps = cluster_max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        save_steps=save_steps,
        logging_steps=logging_steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to=report_to
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # Create a trainer for this cluster
    trainer = SFTTrainer(
        model=model,
        train_dataset=cluster_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )

    # Train the model for this cluster
    trainer.train()

    # Save the trained model
    trainer.model.save_pretrained(new_model_name)
    print(f'Saved {new_model_name}')
    count += 1



