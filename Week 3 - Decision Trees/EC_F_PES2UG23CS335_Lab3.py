# lab.py
import torch

def get_entropy_of_dataset(tensor: torch.Tensor):
    """
    Calculate the entropy of the entire dataset.
    Formula: Entropy = -Σ(p_i * log2(p_i)) where p_i is the probability of class i

    Args:
        tensor (torch.Tensor): Input dataset as a tensor, where the last column is the target.

    Returns:
        float: Entropy of the dataset.
    """
    # TODO: Implement this function
    # Used to compute how impure the dataset is

    target = tensor[:, -1]  # takes the values of the last column - class label
    values, counts = torch.unique(target, return_counts=True) # finds all the unique class labels and count of each
    probabilities = counts.float() / target.shape[0] # converts this count to probabilities

    entropy = -torch.sum(probabilities * torch.log2(probabilities))
    return entropy.item()
    raise NotImplementedError


def get_avg_info_of_attribute(tensor: torch.Tensor, attribute: int):
    """
    Calculate the average information (weighted entropy) of an attribute.
    Formula: Avg_Info = Σ((|S_v|/|S|) * Entropy(S_v)) where S_v is subset with attribute value v.

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
        attribute (int): Index of the attribute column.

    Returns:
        float: Average information of the attribute.
    """
    # TODO: Implement this function
    # Tells us how much uncertainity remains if we split on this attribute
    total_len = tensor.shape[0] # the total number of rows in the dataset
    values, counts = torch.unique(tensor[:, attribute], return_counts=True) # find all unique values for a specific attribute and the count of each

    avg_info = 0.0
    for v, count in zip(values, counts):
        subset = tensor[tensor[:, attribute] == v] # for each possible value, create a subset of the dataset containing only the rows with that attribute value
        subset_entropy = get_entropy_of_dataset(subset) # calculate the entropy of that subset
        avg_info += (count.item() / total_len) * subset_entropy # each subset entropy is then weighted by its proportion in the dataset

    return avg_info
    raise NotImplementedError


def get_information_gain(tensor: torch.Tensor, attribute: int):
    """
    Calculate Information Gain for an attribute.
    Formula: Information_Gain = Entropy(S) - Avg_Info(attribute)

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.
        attribute (int): Index of the attribute column.

    Returns:
        float: Information gain for the attribute (rounded to 4 decimals).
    """
    # TODO: Implement this function
    # Calculate the information gain for a specified attribute
    total_entropy = get_entropy_of_dataset(tensor) # total entropy of the entire dataset
    avg_info = get_avg_info_of_attribute(tensor, attribute) # average entropy after splitting on that specific attribute
    info_gain = total_entropy - avg_info # computes the information gain
    return round(info_gain, 4) 
    raise NotImplementedError


def get_selected_attribute(tensor: torch.Tensor):
    """
    Select the best attribute based on highest information gain.

    Returns a tuple with:
    1. Dictionary mapping attribute indices to their information gains
    2. Index of the attribute with highest information gain
    
    Example: ({0: 0.123, 1: 0.768, 2: 1.23}, 2)

    Args:
        tensor (torch.Tensor): Input dataset as a tensor.

    Returns:
        tuple: (dict of attribute:index -> information gain, index of best attribute)
    """
    # TODO: Implement this function
    num_attributes = tensor.shape[1] - 1  # number of attributes is the number of columns - 1 (last column is the class label)
    info_gains = {} # stores a dictionary of the attribute indices and their information gains

    for attr in range(num_attributes):
        info_gains[attr] = get_information_gain(tensor, attr)

    best_attr = max(info_gains, key=info_gains.get) # finds the attribute with the highest information gain for the best split
    return info_gains, best_attr
    raise NotImplementedError
