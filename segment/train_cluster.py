import os
import math
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
    TrainerCallback,
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

model_name = "meta-llama/Llama-2-7b-hf" # Also try "mistralai/Mistral-7B-v0.1"
dataset = "instruct"
cluster = "narval"

num_clusters = 4  # Adjust the number of clusters as needed
resume_from_cluster = 0 # zero-indexed, resume from last incomplete run
resume_from_checkpoint = None
cluster_strategy = "embeddings"

attention_only = True
layer_config = "att" if attention_only else "lin"

# Based on experimentation, 32/16 seem to be the ideal rank/alpha settings for llama2 and open-instruct
lora_r = 32 # Rank equal to alpha is equivalent to scaling weights by 1
lora_alpha = 16 # Keep this fixed while changing the rank
lora_dropout_factor = 0
lora_dropout = lora_dropout_factor * 0.1

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
    
new_model = f"llama-2-7b-{dataset}-lora-{layer_config}-d{lora_dropout_factor}-r{lora_r}-a{lora_alpha}"

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

report_to = "wandb"
output_dir = new_model
num_train_epochs = 3
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
# Empirically, this seems to harm performance. Need to re-run experiments and verify.

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

def cluster_embeddings(dataset, num_clusters, embeddings):    
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

def cluster_tfid(dataset, num_clusters):    
    # Extract text data (adjust the key 'text' if your dataset has a different text field)
    texts = [entry['text'] for entry in dataset]

    # Data Preparation and Feature Extraction
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(texts)

    # Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(X)

    # Assigning cluster labels to each text
    cluster_labels = kmeans.labels_

    # Creating a DataFrame for easier visualization
    clustered_data = pd.DataFrame({'text': texts, 'cluster': cluster_labels})

    return clustered_data

if cluster_strategy == "embeddings":
    embeddings = np.load(f'{dataset}_embeddings.npy')
    clustered_data = cluster_embeddings(train_dataset, num_clusters, embeddings)
    print(f'Created {num_clusters} clusters from precomputed embeddings file: {dataset}_embeddings.npy')
elif cluster_strategy == "tfid":
    clustered_data = cluster_tfid(train_dataset, num_clusters)

unique_clusters = clustered_data['cluster'].unique()
cluster_datasets = {}

# Loop through each cluster and create datasets
for cluster_label in unique_clusters:
    cluster_df = clustered_data[clustered_data['cluster'] == cluster_label]
    cluster_datasets[f"cluster_{cluster_label}"] = Dataset.from_pandas(cluster_df)
    
# Sort the cluster numerically
cluster_datasets = dict(sorted(cluster_datasets.items()))  

# Create custom checkpoint callbacks
class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        kwargs["model"].save_pretrained(checkpoint_path)

        if "pytorch_model.bin" in os.listdir(checkpoint_path):
            os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))

class SaveEpochCallback(TrainerCallback):
    def __init__(self, save_path):
        self.save_path = save_path

    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = state.epoch
        model_save_path = f"{self.save_path}/epoch_{math.ceil(epoch)}"
        kwargs['model'].save_pretrained(model_save_path)
        print(f"Model saved to {model_save_path} at the end of epoch {epoch}")
    
count = 1
total = len(cluster_datasets.items())

for cluster_label, cluster_dataset in islice(cluster_datasets.items(), resume_from_cluster, total, 1):   
    
    # The QLoRA paper uses a batch size of 16 for a total of 1875 steps for Guanaco (~10K)
    # Which means the model sees 30K examples, or each example 3 times
    # However, Sebastian Raschka's experiments suggest multiple iterations over a dataset harms performance
    dataset_iterations = 1
    cluster_size = len(cluster_dataset)
    cluster_max_steps = dataset_iterations * (cluster_size // per_device_train_batch_size)
    new_model_name = f"llama-2-7b-{dataset}_lora-{layer_config}-d{lora_dropout_factor}-r{lora_r}-a{lora_alpha}-{num_clusters}_{cluster_label}"
    
    # print(f'{count}/{total} Training {cluster_label} for {cluster_max_steps} steps...')
    print(f'({count}/{total}) Training {new_model_name} for {num_train_epochs} epochs...')

    # Initialize both callbacks
    save_model_callback = SaveEpochCallback(new_model_name)
    peft_saving_callback = PeftSavingCallback()
    callbacks = [save_model_callback, peft_saving_callback]  
    
    # Set training parameters
    # Just going to train by epoch instead, easier for experimentation
    training_arguments = TrainingArguments(
        resume_from_checkpoint=resume_from_checkpoint,
        output_dir=new_model_name,
        num_train_epochs=num_train_epochs, # Compare results across epochs
        # max_steps = cluster_max_steps, # Need to investigate how many steps or epochs to train for
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
        callbacks=callbacks,
    )

    # Train the model for this cluster
    checkpoint = None
    if training_arguments.resume_from_checkpoint is not None:
        checkpoint = training_arguments.resume_from_checkpoint
        print(f'Resuming {cluster_label} model training from {resume_from_checkpoint}')
    else:
        print(f'Starting new training run for model: {new_model}')
    trainer.train(resume_from_checkpoint=checkpoint)
    count += 1



