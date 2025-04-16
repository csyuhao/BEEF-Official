from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T


def load_dataset(dataset_name):
    """
    Load the dataset based on the provided name.
    Args:
        dataset_name (str): Name of the dataset to load.
    Returns:
        tuple: A tuple containing the loaded dataset and the number of classes.
    """
    trans = T.Compose([
        T.ToUndirected(),
    ])

    if dataset_name in ("PROTEINS", "ENZYMES", "MUTAG"):
        dataset = TUDataset(root=r'data/', name=dataset_name, transform=trans)
    else:
        raise ValueError('Invalid dataset name.')
    return dataset, dataset.num_classes
