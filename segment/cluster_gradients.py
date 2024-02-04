import os
import glob
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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
    DataCollatorForLanguageModeling
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
from tqdm.auto import tqdm

from sklearn.decomposition import IncrementalPCA
from joblib import dump, load
import gc
from functools import partial

# from cuml.decomposition import IncrementalPCA
# import cupy as cp
# import cupyx

def preprocess_instruct(examples):
    # Concatenate 'prompt' and 'completion' fields
    texts = [prompt + " " + completion for prompt, completion in zip(examples['prompt'], examples['completion'])]
    return {'text': texts}  
    
    
    
### TOKENIZATION ####

def tokenize_function(examples, tokenizer, max_length):
    return tokenizer(
        examples["text"], # Guanaco dataset only has a text field
        # example["prompt"],
        # example["completion"],
        truncation=True,       # truncate to the model's max length
        max_length=max_length,        # max length for the tokens
        padding="max_length",  # add padding to the tokens
        return_tensors="pt"    # return PyTorch tensors
    )

def shift_labels_right(examples):
    examples['labels'] = examples['input_ids'].copy()
    examples['labels'] = [x[1:] + [-100] for x in examples['labels']]
    return examples
    
def get_tokenized_dataloader(tokenized_dataset, tokenizer, batch_size):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Set to False for CLM
    )
    dataloader = DataLoader(
        tokenized_dataset, 
        batch_size=batch_size, 
        collate_fn=data_collator 
    )
    return dataloader

    
    
### GRADIENTS ###

def aggregate_gradients_for_batch(batch, model, max_length=None):
    batch = {k: v.to("cuda") for k, v in batch.items()}
    model.zero_grad()
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    # Collect gradients for the entire batch from the last layer
    gradients_list = []
    batch_gradients = []
    for name, parameter in model.named_parameters():
        if "lm_head" in name and parameter.requires_grad and parameter.grad is not None:
            batch_grads_float32 = parameter.grad.view(-1).to(dtype=torch.float32)
            batch_gradients.append(batch_grads_float32)
    # Concatenate the gradients for the current example and keep on GPU
    gradients_list.append(torch.cat(batch_gradients))

    # Stack the gradients, detach and move the entire tensor to CPU at once
    all_gradients = torch.stack(gradients_list).detach().cpu().numpy()
    return all_gradients


##### CHECKPOINTING ######

def manage_ipca_checkpoints(ipca_model_name, step_count, ipca, max_checkpoints=10):

    checkpoint_pattern = f"{ipca_model_name}_*.joblib"
    existing_checkpoints = glob.glob(checkpoint_pattern)
    sorted_checkpoints = sorted(existing_checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    
    while len(sorted_checkpoints) >= max_checkpoints:
        os.remove(sorted_checkpoints.pop(0))  # Remove the oldest checkpoint
    
    # Save the new checkpoint
    last_saved_pca = f"{ipca_model_name}_{step_count}.joblib"
    dump(ipca, last_saved_pca)
    
    # Add the new checkpoint to the list
    sorted_checkpoints.append(last_saved_pca)
    
    print(f"{step_count}: saving PCA at {last_saved_pca}. There are {len(sorted_checkpoints)} checkpoints saved.")

    
#### CLUSTERING #########

def cluster_gradients(dataset, num_clusters, batch_size, max_length, compute_pca=True, ipca_checkpoint_step=None):

    # For PCA, n_components must be less or equal to batch_size
    n_components = batch_size
    
    # Load the model and tokenizer
    model_name = "meta-llama/Llama-2-7b-hf"  # replace with your model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    tokenized_dataset = dataset.map(
        partial(tokenize_function, tokenizer=tokenizer, max_length=max_length), 
        batched=True,
    )
    tokenized_dataset = tokenized_dataset.map(
        lambda examples: {'input_ids': examples['input_ids'], 'attention_mask': examples['attention_mask']},
        batched=True,
        remove_columns=tokenized_dataset.column_names  # This removes all columns except the ones specified above
    )
    tokenized_dataset = tokenized_dataset.map(shift_labels_right, batched=True)
    print(f"Tokenized dataset length: {len(tokenized_dataset)}")

    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    debug_mode = False
    grad_batching = "aggregate"
      
    
    if debug_mode:
        dataloader = get_tokenized_dataloader(tokenized_dataset, tokenizer, batch_size)
        iterator = iter(dataloader)

        for i in range(batch_size):
            batch = next(iterator)
            batch_gradients = aggregate_gradients_for_batch(batch,model, max_length)
            gradient_buffer.append(batch_gradients)

        gradient_list = np.vstack(gradient_buffer)
        print(f"gradient_list (batch): {gradient_list.shape}")
        ipca.partial_fit(gradient_list)
        gradient_buffer = [] 

        example_batch_size = 1
        dataloader = get_tokenized_dataloader(tokenized_dataset, tokenizer, example_batch_size)
        iterator = iter(dataloader)
        batch = next(iterator)
        example_gradients = aggregate_gradients_for_batch(batch, model, max_length)

        gradient_buffer.append(example_gradients)
        gradient_list = np.vstack(gradient_buffer)
        print(f"gradient_list (reduced example): {gradient_list.shape}")
        reduced_gradients_batch = ipca.transform(gradient_list)

        del gradient_list  # Delete the variable holding the tensor
        del gradient_buffer
        torch.cuda.empty_cache()  # Release GPU memory
        gc.collect()
        gradient_buffer = []  # Clear the buffer
        
    gradient_shape = batch_gradients.shape if debug_mode else (batch_size, 131072000)
    reduced_shape = reduced_gradients_batch.shape if debug_mode else (1, n_components)
    gradient_buffer = []
        
    dataloader = get_tokenized_dataloader(tokenized_dataset, tokenizer, batch_size)
    iterator = iter(dataloader)
    
    step_count = 0
    total_steps = len(iterator)
    
    num_fits = 0
    save_every = 6 # Save after every X fits
    
    gradient_buffer = []
    gradient_buffer_size = batch_size  # The desired buffer size
    expected_batch_size = batch_size
    
    if ipca_checkpoint_step is None:
        load_ipca_checkpoint = False
    else:
        load_ipca_checkpoint = True
        
    ipca_model_name = f"bs{batch_size}_ml{max_length}_lmhead_ipca" # ["ipca_model", "b1_lmhead_ipca"]
    # ipca_checkpoint_step = 1231
    ipca_checkpoint = f"{ipca_model_name}_{ipca_checkpoint_step}.joblib" 
    
    print(f"""
        Starting Incremental PCA training with the following configs, saving after every {save_every} fits.

        Compute IPCA: {compute_pca}
        Load IPCA Checkpoint: {load_ipca_checkpoint}
        IPCA Checkpoint Step: {ipca_checkpoint_step}
        Num Components: {n_components}
        Batch Size: {batch_size}
        Gradient Buffer Size: {gradient_buffer_size}
        Gradient Shape: {gradient_shape}
        Reduced Gradient Shape: {reduced_shape}
        Gradient Batching: {grad_batching}
    """)
    
    if load_ipca_checkpoint:
        ipca = load(ipca_checkpoint)
        start_batch_index = ipca_checkpoint_step
        print(f"Loaded IPCA checkpoint from {ipca_checkpoint}.")
    else:
        start_batch_index = 0

    if compute_pca: # Process gradients in batches and accumulate in buffer
        for i in tqdm(range(0, len(dataloader)), desc='Estimating PCA model from gradients...'):

            batch = next(iterator)
            step_count += 1
            if (load_ipca_checkpoint and step_count > ipca_checkpoint_step) or not load_ipca_checkpoint:                   

                if batch['input_ids'].shape[0] < expected_batch_size:
                    print(f"Skipping a batch with size {len(batch)} which is smaller than the expected size {expected_batch_size}.")
                    continue
                else:
                    batch_gradients = aggregate_gradients_for_batch(batch, model, max_length)
                    gradient_buffer.append(batch_gradients) # Append batch_size (8) gradients to the buffer

                    # If buffer is full (8 gradients from a total of 64 examples) then do a partial fit
                    if len(gradient_buffer) == gradient_buffer_size:
                        gradient_list = np.vstack(gradient_buffer) 
                        ipca.partial_fit(gradient_list)
                        num_fits += 1
                        
                        # TODO: save checkpoint after partial fits so examples are not missed
                        if num_fits % save_every == 0:
                            manage_ipca_checkpoints(ipca_model_name, step_count, ipca)
                        
                        del gradient_list  
                        del gradient_buffer
                        torch.cuda.empty_cache()  
                        gc.collect()
                        gradient_buffer = []  
                        torch.cuda.empty_cache() 
                        gc.collect()


        print(f"{step_count}/{total_steps}: saving PCA at ipca_model_{step_count}.joblib")
        last_saved_pca = f"{ipca_model_name}_{step_count}.joblib"
        dump(ipca, last_saved_pca)

    gradient_buffer = []
    reduced_gradients = []
    example_batch_size = 1 # Get gradient features for 1 example at a time for clustering
    
    dataloader = get_tokenized_dataloader(tokenized_dataset, tokenizer, example_batch_size)
    iterator = iter(dataloader)
    num_examples = len(dataloader)
    
    for i in tqdm(range(0, num_examples), desc='Reducing gradients using fitted PCA...'):       
        batch = next(iterator)
        example_gradients = aggregate_gradients_for_batch(batch, model, max_length)
            
        gradient_buffer.append(example_gradients)
        gradient_list = np.vstack(gradient_buffer)
        reduced_gradients_batch = ipca.transform(gradient_list)
        reduced_gradients.append(reduced_gradients_batch)
        gradient_buffer = []
        
    stacked_gradients = np.stack(reduced_gradients, axis=0)
    stacked_gradients = stacked_gradients.squeeze(axis=1)
    np.save(f"reduced_gradients_stack_bs{batch_size}.npy", stacked_gradients) # Save reduced gradients for different num clusters later
    print(f"Saved reduced gradients with shape {stacked_gradients.shape} to reduced_gradients_stack.npy")
    
    # Cluster the gradient features
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    kmeans.fit(stacked_gradients)
    cluster_labels = kmeans.labels_

    # Create the clustered dataset
    clustered_data = pd.DataFrame({'text': dataset['text'][:num_examples], 'cluster': cluster_labels})
    return clustered_data


# Function to fit IncrementalPCA and calculate cumulative explained variance
def cumulative_explained_variance(X, n_components, batch_size):
    ipca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    ipca.fit(X)
    explained_variance_ratio = ipca.explained_variance_ratio_
    cumulative_variance_ratio = cp.cumsum(explained_variance_ratio).get()
    return cumulative_variance_ratio, ipca

# Function to find the number of components that explain at least the target variance
def find_n_components(X, target_variance, batch_size):
    n_components = 2  # start with 2 components, or a reasonable guess
    cumulative_variance_ratio, ipca = cumulative_explained_variance(X, n_components, batch_size)
    
    # Increase n_components until target_variance is met
    while cumulative_variance_ratio[-1] < target_variance and n_components < X.shape[1]:
        n_components += 1
        cumulative_variance_ratio, ipca = cumulative_explained_variance(X, n_components, batch_size)
    
    return n_components, ipca       









def main(cluster_name, dataset, num_clusters, num_components, max_length, test_pca=False, compute_pca=True, ipca_checkpoint_step=None):
    
    model_name = "meta-llama/Llama-2-7b-hf" # Also try "mistralai/Mistral-7B-v0.1"
    cluster = cluster_name
 
    cluster_strategy = "gradients"
    resume_from_cluster = 0
    

    if cluster == "cedar":
        if dataset == "guanaco":
            dataset_name = "timdettmers/openassistant-guanaco"
            train_dataset = load_dataset(dataset_name, split="train")
        elif dataset == "instruct":
            dataset_name = "monology/VMware-open-instruct-higgsfield"
            train_dataset = load_dataset(dataset_name, split="train")
            train_dataset = train_dataset.map(preprocess_instruct, batched=True)
        print(f"Training dataset set to {dataset_name} from hugging face")

    if cluster == "narval":
        if dataset == "guanaco":
            dataset_path = "/scratch/alif/timdettmers___json/timdettmers--openassistant-guanaco-c93588435bc90172/0.0.0/fe5dd6ea2639a6df622901539cb550cf8797e5a6b2dd7af1cf934bed8e233e6e/json-train.arrow"
            train_dataset = Dataset.from_file(dataset_path)
        elif dataset == "instruct":
            dataset_path = '/scratch/alif/monology___v_mware-open-instruct-higgsfield/default/0.0.0/622a7cf65a222fcb/v_mware-open-instruct-higgsfield-train.arrow'
            train_dataset = Dataset.from_file(dataset_path)
            train_dataset = train_dataset.map(preprocess_instruct, batched=True)
        print(f"Training dataset set to: {dataset} from local path: {dataset_path}")
    
    if not test_pca:
        print(f"""
            Fitting PCA with the following configs:
            Checkpoint Step: {ipca_checkpoint_step}
            Compute PCA: {compute_pca}
            Num Clusters: {num_clusters}
            Num Components: {num_components}
            Max Seq Length: {max_length}
        """)
        
        clustered_data = cluster_gradients(train_dataset, num_clusters, num_components, max_length, compute_pca, ipca_checkpoint_step)
        save_path = f"gradient_clusters_{num_clusters}.csv" 
        clustered_data.to_csv(save_path)
        print(f"Clustered data using model gradients and saved dataframe to {save_path}")

        unique_clusters = clustered_data['cluster'].unique()
        cluster_datasets = {}

        # Loop through each cluster and create datasets
        for cluster_label in unique_clusters:
            cluster_df = clustered_data[clustered_data['cluster'] == cluster_label]
            cluster_datasets[f"cluster_{cluster_label}"] = Dataset.from_pandas(cluster_df)

        # Sort the cluster numerically
        cluster_datasets = dict(sorted(cluster_datasets.items()))  

        count = 1
        total = len(cluster_datasets.items())

        for cluster_label, cluster_dataset in islice(cluster_datasets.items(), resume_from_cluster, total, 1):   

            print(f'({count}/{total}) Saving {cluster_label} for {dataset} dataset...')
            print(cluster_dataset[0])
            dataset_length = len(cluster_dataset)    
            dataset_name = f"data/{dataset}_pca{num_components}_{num_clusters}_{cluster_label}_{dataset_length}.json"
            cluster_dataset.to_json(dataset_name)
            count += 1
    else:
        print(f"Finding ideal number of (incremental) PCA components for gradient reduction.")
        
        # Load the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        tokenized_dataset = train_dataset.map(
            partial(tokenize_function, tokenizer=tokenizer, max_length=max_length), 
            batched=True,
        )
        tokenized_dataset = tokenized_dataset.map(
            lambda examples: {'input_ids': examples['input_ids'], 'attention_mask': examples['attention_mask']},
            batched=True,
            remove_columns=tokenized_dataset.column_names  # This removes all columns except the ones specified above
        )
        tokenized_dataset = tokenized_dataset.map(shift_labels_right, batched=True)
        print(f"Tokenized dataset length: {len(tokenized_dataset)}")

        # Start with 2 components        
        num_batches = 3
        num_components = num_components
        buffer_size = num_components
        dataloader_batch_size = 8
        dataloader = get_tokenized_dataloader(tokenized_dataset, tokenizer, dataloader_batch_size)
        iterator = iter(dataloader)
        
        target_variance = 0.95
        ipca = IncrementalPCA(n_components=num_components)
        gradient_buffer = []
        for i in tqdm(range(num_batches), desc=f"Performing partial fits for {num_batches} batches of {num_components} gradients"):
            for i in range(buffer_size):
                batch = next(iterator)
                batch_gradients = aggregate_gradients_for_batch(batch, model, max_length)
                gradient_buffer.append(batch_gradients)

            gradient_list = np.vstack(gradient_buffer)
            ipca.partial_fit(gradient_list)
            
            del gradient_list  # Delete the variable holding the tensor
            del gradient_buffer
            torch.cuda.empty_cache()  # Release GPU memory
            gc.collect()
            gradient_buffer = []  # Clear the buffer
        
        explained_variance_ratio = ipca.explained_variance_ratio_
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)[-1]
        
        # print(f"Number of components to explain {target_variance*100}% of variance: {num_components}")
        print(f"IPCA explained variance ratio: {explained_variance_ratio}")
        print(f"Cumulative explained variance ratio: {cumulative_variance_ratio}")


if __name__ == "__main__":
    
    num_clusters = 2  # Adjust the number of clusters as needed
    num_components = 12 # Number of PCA components
    max_length = 128 # Max sequence length for each input
    
    ipca_checkpoint_step = 1970
    test_pca = True
    compute_pca = False
    
    cluster_name = "narval"
    dataset = "guanaco"
    
    main(cluster_name, dataset, num_clusters, num_components, max_length, test_pca=test_pca, compute_pca=compute_pca, ipca_checkpoint_step=ipca_checkpoint_step)
    
