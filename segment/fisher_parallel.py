# Trainable param regex: https://github.com/r-three/mats/blob/a7165f84cff194596465b50c49a49bcd4dbd0fbe/src/model/load_model.py#L63

# Use conjugate gradient calculation from scipy
import os
import re
from scipy.sparse.linalg import LinearOperator, cg # for conjugate gradient methods

from safetensors import safe_open # for checkpoint loading
from safetensors.torch import load_model, save_model, load_file
from safetensors.torch import load as load_safetensors

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    TrainingArguments,
    pipeline,
    logging,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, PeftModel

import torch
from torch.utils import data
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm 
from itertools import islice
import numpy as np

# FISHER MERGING METHODS

def compute_diag_fisher(model, param_regex):
    example_fisher = {}
    for param_name, param in model.named_parameters():
        if re.fullmatch(param_regex, param_name) and param.requires_grad:
            # Ensure that the gradient computation is done on the CPU
            # grad = param.grad.cpu() if param.grad is not None else None
            grad = param.grad if param.grad is not None else None
            if grad is not None:
                # Initialize the example_fisher entry on the CPU
                if param_name not in example_fisher:
                    example_fisher[param_name] = torch.zeros_like(param.data)
                # Perform the square operation on CPU and accumulate
                example_fisher[param_name] += torch.square(grad)

    return example_fisher
    

def get_model_fisher(checkpoint_path, dataset, tokenizer, fisher_path):

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto", # Parallelize across GPUs
    )

        
    for param in model.parameters():
        param.requires_grad = True
    
    req_grad_count = 0
    lora_count = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f"Parameter {name} does not require grad.")
            req_grad_count += 1
        if 'lora' in name:
            print(f"Parameter {name} is a LoRA parameter.")
            lora_count += 1
            
    if req_grad_count == 0:
        print("All parameters require grad.")        
    if lora_count == 0:
        print("All LoRA parameters have been merged.")
        
    model.eval()
    stored_fisher = {}
    
    def update_fisher(example_fisher):
        for param_name, value in example_fisher.items():
            if param_name not in stored_fisher:
                stored_fisher[param_name] = value
            else:
                stored_fisher[param_name] += value
    rank = 0
    batch_size = 1 
    
    dataloader = get_tokenized_dataloader(dataset, tokenizer, batch_size)
    
    num_samples = 0
    param_regex = ".*"
    
    print(f'Calculating losses for {checkpoint_path}...')
    for batch in tqdm(islice(dataloader, 1000), total=1000):
        batch = {k: v.to("cuda") for k, v in batch.items()}
        
        outputs = model(**batch)
        loss = outputs.loss
        log_prob = -loss
        log_prob.backward()

        # Compute the per-example fisher and update total fisher
        with torch.no_grad():
            example_fisher = compute_diag_fisher(model, param_regex)
            update_fisher(example_fisher)

        num_samples += 1
        model.zero_grad()
            
    with torch.no_grad():
        stored_fisher = normalize_metadata(stored_fisher, num_samples)
        
    torch.save(detach_metadata(stored_fisher), fisher_path)
    
    
def conjugate_gradient_forward(
    sum_fisher_matrices, 
    sum_fisher_times_weight, 
    init_model,
    all_param_names,
    num_iters
):
    final_model = {}
    
    for param_name in tqdm(all_param_names):
        weight_shape = sum_fisher_times_weight[param_name].shape
        
        def matrix_vector_product(v):
            v_weight_torch = from_numpy(v).reshape(weight_shape).float()
            matrix_vector = torch.mul(sum_fisher_matrices[param_name], v_weight_torch)
            return matrix_vector.flatten().cpu().numpy()
        
        b = sum_fisher_times_weight[param_name].detach().flatten()
        A = LinearOperator(
            (weight_shape.numel(), weight_shape.numel()), matvec=matrix_vector_product
        )
        
        if init_model is not None:
            init_x0 = init_model[param_name].detach().cpu().numpy().flatten()
            x_final, exit_code = conjugate_gradient(A, b, x0=init_x0, maxiter=num_iters)
            initial_error = np.linalg.norm(matrix_vector_product(init_x0) - b.numpy())
        else:
            x_final, exit_code = conjugate_gradient(A, b, maxiter=num_iters)
        
        final_error = np.linalg.norm(matrix_vector_product(x_final) - b.numpy())
        
        final_weight = torch.tensor(x_final).reshape(weight.shape)
        final_model[param_name] = final_weight
        
        return final_model
    

    
# DATASET AND CHECKPOINT LOADING

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

def load_checkpoints(pretrained_model, num_clusters, device, target_epoch=None):
    """
    Load model cluster checkpoints to merge.
    """

    checkpoint_dict = {}
    base_path = "./"

    for cluster_idx in range(num_clusters):
        cluster_name = f"{pretrained_model}_cluster_{cluster_idx}"
        cluster_path = os.path.join(base_path, cluster_name)

        epochs = [d for d in os.listdir(cluster_path) if d.startswith('epoch')]
        if not epochs:
            raise FileNotFoundError(f"No epochs found in {cluster_path}")

        latest_epoch = sorted(epochs, key=lambda x: int(x.split('_')[1]), reverse=True)[0]
        epoch = latest_epoch if target_epoch is None else target_epoch
        
        checkpoint_file = os.path.join(cluster_path, epoch, 'model/')
        checkpoint_dict[cluster_name] = checkpoint_file

    return checkpoint_dict

def combine_safetensors(file_paths):
    """
    Load multiple SafeTensor files and combine their state dictionaries.
    """
    combined_state_dict = {}

    for file_path in file_paths:
        # Load the state dict from the current SafeTensor file
        with open(file_path, "rb") as f:
            data = f.read()
        f.close()
            
        state_dict = load_safetensors(data)

        # Check for key conflicts before updating
        intersecting_keys = combined_state_dict.keys() & state_dict.keys()
        if intersecting_keys:
            raise ValueError(f"Conflicting keys found when loading {file_path}: {intersecting_keys}")

        # Update the combined state dictionary
        combined_state_dict.update(state_dict)

    return combined_state_dict


# REGEX FUNCTIONS

def get_cluster(checkpoint):
    pattern = re.compile(r"cluster_\d+")
    matches = pattern.findall(checkpoint)
    return matches[-1] if matches else None

def get_model_safetensors(directory):
    pattern = re.compile(r'model-\d+-of-\d+\.safetensors$')
    model_files = []
    for filename in os.listdir(directory):
        if pattern.match(filename) and not filename.startswith('adapter_model'):
            model_files.append(os.path.join(directory, filename))

    model_files.sort()
    print(model_files)

    return model_files

def extract_epoch(checkpoint_path):
    epoch_regex = re.compile(r'epoch_(\d+)')
    match = epoch_regex.search(checkpoint_path)
    if match:
        return int(match.group(1))
    else:
        return None    

        

# MODEL OPERATIONS

def map_params(model_params, map_fn):
    new_params = {}
    for param_name, param_value in model_params.items():
        new_params[param_name] = map_fn(param_value)
    return new_params

def scale(model_params, scaler):
    scale_fn = lambda x: x * scaler
    scaled_model = map_params(model_params, scale_fn)
    return scaled_model

def reduce_params(model_params, reduce_fn):
    param_values = zip(*list(map(lambda x: x.values(), model_params)))
    param_names = model_params[0].keys()
    
    new_params = {}
    for param_name, param_value in zip(*[param_names, param_values]):
        new_params[param_name] = reduce_fn(torch.stack(list(param_value), dim=0))
    return new_params

def scale_and_sum(model_params, model_lambda):
    sum_fn = lambda parameters: torch.sum(parameters * model_lambda, dim=0)
    summed_model = reduce_params(model_params, sum_fn)
    return summed_model

def pairwise_param_map(params_a, params_b, map_fn):
    a_params = params_a.keys()
    b_params = params_b.keys()
    
    new_params = {}
    
    for param_name in a_params:
        new_params[param_name] = map_fn(params_a[param_name], params_b[param_name])
    return new_params

def element_wise_multiply(params_a, params_b):
    element_wise_mul = lambda x, y: torch.mul(x, y)
    element_wise_mul_model = pairwise_param_map(params_a, params_b, element_wise_mul)
    return element_wise_mul_model

def divide(params_a, params_b):
    divide_fn = lambda x, y: x / y
    divide_model = pairwise_param_map(
        params_a, params_b, divide_fn
    )
    return divide_model

def set_minimum(model_parameters, epsilon):
    """
    Set the minimum of the parameters to be epsilon. For any value less than epsilon,
    replace with epsilon
    """
    new_modelParameters = {}
    for parameter_name, parameter in model_parameters.items():
        new_parameter = parameter.clone()
        new_parameter[new_parameter < epsilon] = epsilon
        new_modelParameters[parameter_name] = new_parameter
    return new_modelParameters

def normalize_metadata(stored_metadata, count):
    normalized_metadata = {}
    for parameter_name, parameter in stored_metadata.items():
        normalized_metadata[parameter_name] = parameter / count
    return normalized_metadata

def detach_metadata(stored_metadata):
    detached_metadata = {}
    for parameter_name, parameter in stored_metadata.items():
        detached_metadata[parameter_name] = parameter.detach().contiguous().cpu()
    return detached_metadata


# HUGGING FACE DATA PRE-PROCESSING

def preprocess_instruct(examples):
    # Concatenate 'prompt' and 'completion' fields
    texts = [prompt + " " + completion for prompt, completion in zip(examples['prompt'], examples['completion'])]
    return {'text': texts}

# Remove 'prompt' and 'completion' keys from the dataset
def remove_unnecessary_columns(example):
    return {
        "input_ids": example["input_ids"],
        "attention_mask": example["attention_mask"]
    }

def shift_labels_to_the_right(examples):
    examples['labels'] = examples['input_ids'].copy()
    examples['labels'] = [x[1:] + [-100] for x in examples['labels']]
    return examples

# Tokenize the dataset
def tokenize_function(example):
    return tokenizer(
        example["prompt"],
        example["completion"],
        truncation=True,       # truncate to the model's max length
        max_length=128,        # max length for the tokens
        padding="max_length",  # add padding to the tokens
        return_tensors="pt"    # return PyTorch tensors
    )



if __name__ == "__main__":
    
    # COMPUTE FISHERS
    # https://github.com/r-three/mats/blob/main/src/merging/save_metadata/save_fisher.py
    
    dataset = "instruct"
    cluster = "cedar"
    
    compute_fishers = False
    model_lambda_factor = 9
    model_lambda = 0.1 * model_lambda_factor

    if cluster == "cedar":
        if dataset == "guanaco":
            dataset_name = "timdettmers/openassistant-guanaco"
        elif dataset == "instruct":
            dataset_name = "monology/VMware-open-instruct-higgsfield"
        train_dataset = load_dataset(dataset_name, split="train")
        train_dataset = train_dataset.map(preprocess_instruct, batched=True)
        print(f"Training dataset set to {dataset_name} from hugging face")

    if cluster == "narval":
        if dataset == "guanaco":
            dataset_path = "/scratch/alif/timdettmers___json/timdettmers--openassistant-guanaco-c93588435bc90172/0.0.0/fe5dd6ea2639a6df622901539cb550cf8797e5a6b2dd7af1cf934bed8e233e6e/json-train.arrow"
        elif dataset == "instruct":
            dataset_path = '/scratch/alif/monology___v_mware-open-instruct-higgsfield/default/0.0.0/622a7cf65a222fcb/v_mware-open-instruct-higgsfield-train.arrow'
        train_dataset = Dataset.from_file(dataset_path)
        train_dataset = train_dataset.map(preprocess_instruct, batched=True)
        print(f"Training dataset set to: {dataset} from local path: {dataset_path}")
        
    model_name = "meta-llama/Llama-2-7b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Pre-process and tokenize dataset for loss calculations
    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_dataset = tokenized_dataset.map(remove_unnecessary_columns, batched=True)
    tokenized_dataset = tokenized_dataset.map(
        lambda examples: {'input_ids': examples['input_ids'], 'attention_mask': examples['attention_mask']},
        batched=True,
        remove_columns=tokenized_dataset.column_names  # This removes all columns except the ones specified above
    )
    tokenized_dataset = tokenized_dataset.map(shift_labels_to_the_right, batched=True)
    
    # Load checkpoints
    pretrained_model = "llama-2-7b-instruct_lora-att-d0-r32-a16-2"
    num_clusters = 2
    checkpoint_dict = load_checkpoints(pretrained_model, num_clusters, torch.device("cpu"), target_epoch='epoch_1')
    
    if compute_fishers:
        for model_folder in checkpoint_dict.keys():
            print(f'Calculating fisher for {model_folder}...')
            fisher_name = f"{model_folder}_fisher.pt"
            fisher_path = os.path.join('fishers/', fisher_name)
            # model = load_file(checkpoint_dict[model_folder])
            checkpoint_path = checkpoint_dict[model_folder]
            get_model_fisher(checkpoint_path, tokenized_dataset, tokenizer, fisher_path)
            print(f'Saved fisher for {model_folder} at {fisher_path}')
        
        
    # MERGE FISHERS
    # https://github.com/r-three/mats/blob/main/src/merging/diagonal_fisherMerging.py
    
    # Load fishers
    print(f"Beginning model merging process with model lambda {model_lambda}...")
    
    loaded_fishers = {}
    for model_folder in checkpoint_dict.keys():
        fisher_name = f"{model_folder}_fisher.pt"
        fisher_path = os.path.join('fishers/', fisher_name)
        fisher = torch.load(fisher_path, torch.device("cpu"))
        loaded_fishers[model_folder] = fisher
        print(f'Loaded fisher for {model_folder}!')
    print(f"keys for loaded_fishers: {loaded_fishers.keys()}")
    
    checkpoint_fisher_matrices = {}
    
    # The original implementation merges checkpoints by dataset
    # For adapters, merge checkpoints by cluster
    cluster_keys = []
    
    for model_folder, checkpoint_path in checkpoint_dict.items():
        epoch_num = extract_epoch(checkpoint_path)
        cluster = get_cluster(model_folder)
        cluster_keys.append(cluster)
        print(f"Adding checkpoint path for {model_folder} and {cluster}: {checkpoint_path}")
        checkpoint_fisher_matrices[cluster] = {"checkpoint": checkpoint_path}
        
    for model_folder, loaded_fisher in loaded_fishers.items():
        cluster = get_cluster(model_folder)
        if cluster not in checkpoint_fisher_matrices:
            raise ValueError(f"Cluster key {cluster} not found in checkpoint_fisher_matrices")
        checkpoint_fisher_matrices[cluster].update({"fisher": loaded_fisher})
    
    weighted_checkpoint_list = []
    fisher_list = []
    
    for cluster in cluster_keys:
        if 'checkpoint' not in checkpoint_fisher_matrices[cluster] or 'fisher' not in checkpoint_fisher_matrices[cluster]:
            raise KeyError(f"Missing 'checkpoint' or 'fisher' key for cluster {cluster}")
    
    for cluster, checkpoint_fisher_matrix in checkpoint_fisher_matrices.items():
        print(f"cluster name: {cluster}")
        print(f"matrix keys: {checkpoint_fisher_matrix.keys()}")
        
        checkpoint_path = checkpoint_fisher_matrix["checkpoint"]
        checkpoint_model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
        )
        checkpoint = checkpoint_model.state_dict()
        checkpoint = {key: value.to('cpu') for key, value in checkpoint.items()}
        
        print(f'Loaded checkpoint for {cluster}')

        fisher = checkpoint_fisher_matrix["fisher"]
        fisher = set_minimum(fisher, 1e-8)
        print(f'Loaded fisher for {cluster}')
        
        # Scale fishers
        if len(checkpoint_dict) == 2:
            if len(fisher_list) == 0:
                fisher = scale(fisher, model_lambda)
            else:
                assert len(fisher_list) == 1
                fisher = scale(fisher, (1 - model_lambda))
                
        weighted_cp = pairwise_param_map(
            checkpoint, fisher, lambda x, y: x * y
        )
        
        weighted_checkpoint_list.append(weighted_cp)
        fisher_list.append(fisher)
        
    merged_model = divide(
        scale_and_sum(weighted_checkpoint_list, 1),
        scale_and_sum(fisher_list, 1)
    )
    
    merged_filename = pretrained_model + '-merged-ep' + str(epoch_num) + '-ml' + str(model_lambda_factor) + '.pt'
    merged_path = os.path.join('merged_models/', merged_filename)
    torch.save(merged_model, merged_path)
    print(f'Merged models and saved to {merged_path}')