import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.functional import normalize
from functorch import grad, make_functional_with_buffers, vmap

from peft import PeftModel
from tqdm import tqdm

from trak.projectors import BasicProjector, CudaProjector, ProjectionType


def prepare_batch(batch, device=torch.device("cuda:0"))
    """ Move batch to device. """
    for key in batch:
        batch[key] = batch[key].to(device)
        
def prepare_optimizer_state(model, optimizer_state, device):
    """ Get first and second moment estimates for adam optimizer. """
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    avg = torch.cat([optimizer_state[n]["exp_avg"].view(-1) for n in names])
    avg_sq = torch.cat([optimizer_state[n]["exp_avg_sq"].view(-1) for n in names])
    
    avg = avg.to(device)
    avg_sq = avg_sq.to(device)
    return avg, avg_sq


def get_max_saved_index(output_dir: str, prefix="reps") -> int:
    """
    Retrieve the highest index for which the data has been stored
    """
    files = [file for file in os.listdir(output_dir) if file.startswith(prefix)]
    index = [int(file.split(".")[0].split("-")[1]) for file in files] # e.g. output_dir/reps-100.pt
    return max(index) if len(index) > 0 else -1

def get_number_of_params(model):
    """ Make sure that only lora params require grads. """
    if isinstance(model, PeftModel):
        names = [n for n, p in model.named_parameters() if p.requires_grad and "lora" not in n]
        assert len(names) == 0
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
    print(f"Total number of params that require grad: {num_params}")



def get_trak_projector(device: torch.device):
    """  Get trak projectors (see https://github.com/MadryLab/trak for details) """
    try:
        num_sms = torch.cuda.get_device_properties(device.index).multi_processor_count
        import fast_jl
        
        fast_jl.project_rademacher_8(torch.zeros(8, 1_000, device=device), 512, 0, num_sms)
        projector = CudaProjector
        print("Using CudaProjector")
    except:
        projector = BasicProjector
        print("Using BasicProjector")
    return projector



    


def obtain_gradients(model, batch):
    """ Obtain gradients. """
    loss = model(**batch).loss
    loss.backward()
    vectorized_grads = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
    return vectorized_grads

def obtain_sign_gradients(model, batch):
    loss = model(**batch).loss
    loss.backward()
    
    # Instead of concat grads, concat their signs
    vectorized_grad_signs = torch.cat([torch.sign(p.grad).view(-1) for p in model.parameters() if p.grad is not None])
    return vectorized_grad_signs

def obtain_gradients_with_adam(model, batch, avg, avg_sq):
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-08
    
    loss = model(**batch).loss
    loss.backward()
    
    vectorized_grads = torch.cat(p.grad.view(-1) for n, p in model.named_parameters() if p.grad is not None)
    
    # What is this computation doing?
    updated_avg = beta1 * avg + (1 - beta1) * vectorized_grads
    updated_avg_sq = beta2 * avg_sq + (1 - beta2) * vectorized_grads ** 2
    vectorized_grads = updated_avg / torch.sqrt(updated_avg_sq + eps)
    
    return vectorized_grads



def collect_grads(dataloader,
                  model,
                  output_dir,
                  proj_dim: List[int] = [8192],
                  adam_optimizer_state: Optional[dict] = None,
                  gradient_type: str = "adam",
                  max_samples: Optional[int] = None):
    
    """
    Collects gradients from the model during eval and saves them to disk.
    """
    
    torch.random.manual_seed(0)
    
    model_id = 0 # used to draft random seed for projectors
    block_size = 128 # fixed block size for projectors
    
    projector_batch_size = 16
    project_interval = 16 # project every 16 batches
    save_interval = 160 # save every 160 batches
    
    def _project(current_full_grads, projected_grads):
        current_full_grads = torch.stack(current_full_grads).to(torch.float16) # torch.stack concats along new dim (2,3), whereas torch.cat concats along existing dim (6,)
        for i, projector in enumerate(projectors):
            current_projected_grads = projector.project(current_full_grads, model_id=model_id)
            projected_grads[proj_dim[i]].append(current_projected_grads.cpu())
            
    def _save(projected_grads, output_dirs):
        for dim in proj_dim:
            if len(projected_grads[dim]) == 0:
                continue
            projected_grads[dim] = torch.cat(projected_grads[dim])
            
            output_dir = output_dirs[dim]
            outfile = os.path.join(output_dir, f"grads-{count}.pt")
            torch.save(projected_grads[dim], outfile)
            print(f"Saving {outfile}, {projected_grads[dim].shape}", flush=True) # flushes the buffer and displays immediately
            projected_grads[dim] = []
            
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    # prepare optimization states
    if gradient_type == "adam":
        assert adam_optimizer_state is not None
        m, v = prepare_optimizer_state(model, adam_optimizer_state, device) 
        
        
    projector = get_trak_projector(device)
    number_of_params = get_number_of_params(model) 
    
    # initialize a projection for each target proj dim and keep track in a list
    projectors = []
    for dim in proj_dim:
        # Look up a source for further reading on projections / rademacher
        proj = projector(grad_dim=number_of_params,
                         proj_dim=dim,
                         seed=0,
                         proj_type=ProjectionType.rademacher,
                         device=device,
                         dtype=dtype,
                         block_size=block_size,
                         max_batch_size=projector_batch_size) 
        projectors.append(proj)
        
    count = 0
    
    # setup output dir for each dim and keep track in a dictionary
    output_dirs = {}
    for dim in proj_dim:
        output_dir_per_dim = os.path.join(output_dir, f"dim{dim}")
        output_dirs[dim] = output_dir_per_dim
        os.makedirs(output_dir_per_dim, exist_ok=True)
        
    # max saved checkpoint index for each dim
    max_index = min(get_max_saved_index(output_dirs[dim], "grads") for dim in proj_dim) 
    
    # projected gradients
    full_grads = []
    projected_grads = {dim: [] for dim in proj_dim} # initialize dictionary w/ entry for each dim
    
    for batch in tqdm(dataloader, total=len(dataloader))
        prepare_batch(batch) 
        count += 1 
    
        if count <= max_index:
            print("skipping count", count)
            continue
            
        if gradient_type == "adam":
            if count == 1:
                print("Using Adam gradients")
            vectorized_grads = obtain_gradients_with_adam(model, batch, m, v)
        elif gradient_type == "sign":
            if count == 1:
                print("Using Sign gradients") # What are these?
            vectorized_grads = obtain_sign_gradients(model, batch)
        else:
            if count == 1:
                print("Using SGD gradients")
            vectorized_grads = obtain_gradients(model, batch)
            
        full_grads.append(vectorized_grads)
        model.zero_grad()
        
        # compute projections and clear full grad buffer
        if count % project_interval == 0:
            _project(full_grads, projected_grads)
            full_grads = [] 
            
        if count % save_interval == 0:
            _save(projected_grads, output_dirs)
            
        # What does this do??
        if max_samples is not None and count == max_samples:
            break
            
    # project remaining grads
    if len(full_grads) > 0:
        _project(full_grads, projected_grads)
        full_grads = []
        
    for dim in proj_dim:
        _save(projected_grads, output_dirs)
        
    torch.cuda.empty_cache()
    for dim in proj_dim:
        output_dir = output_dirs[dim]
        merge_info(output_dir, prefix="grads", normalize=True)
        merge_info(output_dir, prefix="grads", normalize=False)
        
    print("Finished")
    
    
    
def merge_info(output_dir: str, prefix="reps", normalize=True):
    """ Merge and normalize the representations and gradients into a single file. """
    info = os.listdir(output_dir)
    info = [file for file in info if file.startswith(prefix)]
    
    info.sort(key=lambda x: int(x.split(".")[0].split("-1")[1])) # e.g. reps-100.pt, sort from lowest to highest
    merged_data = []
    
    for file in info:
        data = torch.load(os.path.join(output_dir, file))
        if normalize:
            normalized_data = normalize(data, dim=1)
            merged_data.append(normalized_data)
        else:
            merged_data.append(data)
    merged_data = torch.cat(merged_data, dim=0)
    
    if normalize:
        output_file = os.path.join(output_dir, f"all_orig.pt")
    else:
         output_file = os.path.join(output_dir, f"all_unormalized.pt")
    torch.save(merged_data, output_file)
    print(f"Saving the normalized {prefix} (Shape: {merged_data.shape}) to {output_file}.")
    
