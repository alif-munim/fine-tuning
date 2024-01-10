import torch

def print_parameter_dict(d, indent=0):
    for k, v in d.items():
        indentation = ' ' * indent
        if torch.isnan(v).any():
            print(f"{indentation}{k}: {type(v)}, contains NaN")
        elif not isinstance(v, torch.Tensor):
            print(f"{indentation}{k}: {type(v)}, not a tensor")

def debug_params(model_label, model_params, verbose=False):
    bug_counter = 0
    nan_params = []
    non_tensor_params = []
    for param_name, param_value in model_params.items():
        if isinstance(param_value, tuple):
            if verbose: print(f"WARNING: Tuple found in {model_label} in parameter: {param_name}")
            if verbose: print(param_value)
            bug_counter += 1
            non_tensor_params.append(param_name)
        elif not isinstance(param_value, torch.Tensor):
            if verbose: print(f"WARNING: Non-tensor found in {model_label} in parameter: {param_name}: {type(param_value)}")
            bug_counter += 1
            non_tensor_params.append(param_name)
        elif torch.isnan(param_value).any():
            if verbose: print(f"WARNING: parameter {param_name} contains NaN")
            bug_counter += 1
            nan_params.append(param_name)
    if bug_counter == 0:
        print(f"SUCCESS: {model_label} contains no NaN or non-tensor parameter values")
    else:
        if len(nan_params) > 0:
            print(f"WARNING: NaN param found in {model_label} in parameters (first 10 of {len(nan_params)}): \n{nan_params[:10]}")
            print_parameter_dict(model_params, indent=4)
        elif len(non_tensor_params) > 0:
            print(f"WARNING: Non-tensor param found in {model_label} in parameters (first 10 of {len(non_tensor_params)}): {non_tensor_params[:10]}")
            print_parameter_dict(model_params, indent=4)

def compare_params(dict1, dict2):
    # Find keys that are only in the first dictionary
    unique_to_dict1 = set(dict1.keys()) - set(dict2.keys())
    
    # Find keys that are only in the second dictionary
    unique_to_dict2 = set(dict2.keys()) - set(dict1.keys())
    
    # Report the differences
    if unique_to_dict1 or unique_to_dict2:
        print("The parameter dictionaries do not align.")
        if unique_to_dict1:
            print(f"Keys unique to the first dictionary: {unique_to_dict1}")
        if unique_to_dict2:
            print(f"Keys unique to the second dictionary: {unique_to_dict2}")
        return unique_to_dict1, unique_to_dict2
    else:
        print("The parameter dictionaries align perfectly.")
        return None, None

def pairwise_param_map(params_a, params_b, map_fn, select):
    a_params = params_a.keys()
    b_params = params_b.keys()
    
    new_params = {}
    
    for param_name in a_params:
        if select is not None and select in param_name:
            new_params[param_name] = map_fn(params_a[param_name], params_b[param_name])
        else:
            new_params[param_name] = params_a[param_name]
    return new_params



# params_1 = {'a': torch.tensor([2.0, 4.0]), 'b': torch.tensor([8.0])}
# params_2 = {'a': torch.tensor([1.0, 2.0]), 'b': torch.tensor([4.0])}
# select_param = 'a'

debug_mode = True
weighted_cp_sum = torch.load("weighted_cp_sum.pt")
fisher_sum = torch.load("fisher_sum.pt")

if debug_mode: debug_params("weighted_cp_sum", weighted_cp_sum)
if debug_mode: debug_params("fisher_sum", fisher_sum)
select_param = 'lora'

params_mul = pairwise_param_map(
    weighted_cp_sum, fisher_sum, lambda x, y: x * y, select=select_param
)

params_div = pairwise_param_map(
    weighted_cp_sum, fisher_sum, lambda x, y: x / y, select=select_param
)

# print(f"Result of multiply: {params_mul}")
# print(f"Result of divide: {params_div}")

if debug_mode: debug_params("params_mul", params_mul)
if debug_mode: debug_params("params_div", params_div)

