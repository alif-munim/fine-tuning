import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset, Dataset

from sklearn.cluster import KMeans
from tqdm import tqdm

def preprocess_instruct(examples):
    # Concatenate 'prompt' and 'completion' fields
    texts = [prompt + " " + completion for prompt, completion in zip(examples['prompt'], examples['completion'])]
    return {'text': texts}

dataset = "instruct"
cluster = "narval"

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

def average_pool(last_hidden_states, attention_mask):
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_embeddings_in_batches(texts, model, tokenizer, batch_size=32):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
        batch_texts = texts[i:i + batch_size]
        batch_dict = tokenizer(batch_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(torch.device('cuda'))
        outputs = model(**batch_dict)
        embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        all_embeddings.append(normalized_embeddings.detach().cpu().numpy())
    return np.vstack(all_embeddings)

# Check for CUDA availability
if not torch.cuda.is_available():
    raise EnvironmentError("CUDA not available or GPUs not detected")

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
model = AutoModel.from_pretrained("thenlper/gte-base")

# Utilizing multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = torch.nn.DataParallel(model)

model.to(torch.device('cuda'))  # Move the model to GPU

# Assuming you have a Hugging Face dataset loaded into train_dataset
texts = [entry['text'] for entry in train_dataset]  # Extract texts from the dataset

# Get embeddings and save them
embeddings = get_embeddings_in_batches(texts, model, tokenizer, batch_size=32)
fname = dataset + '_embeddings.npy'
np.save(fname, embeddings)




