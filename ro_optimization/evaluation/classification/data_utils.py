# data_utils.py
import pandas as pd
import torch


def load_celeba_splits(partition_file):
    """
    Load the official CelebA train/val/test splits.
    
    Args:
        partition_file: Path to list_eval_partition.txt
        
    Returns:
        dict with 'train', 'val', 'test' keys containing lists of image names
    """
    # Read partition file
    df = pd.read_csv(partition_file, sep=' ', names=['image', 'partition'], header=None)
    
    # CelebA partition: 0=train, 1=val, 2=test
    train_images = df[df['partition'] == 0]['image'].tolist()
    val_images = df[df['partition'] == 1]['image'].tolist()
    test_images = df[df['partition'] == 2]['image'].tolist()
    
    return {
        'train': train_images,
        'val': val_images,
        'test': test_images
    }


def compute_pos_weight(dataset, max_weight=10.0):
    """
    Compute positive class weights for handling imbalanced data.
    
    Args:
        dataset: Dataset object
        max_weight: Maximum weight to prevent extreme values
        
    Returns:
        Tensor of positive weights for each class
    """
    # Collect all labels
    all_labels = []
    for i in range(len(dataset)):
        item = dataset[i]
        all_labels.append(item['labels'])
    
    all_labels = torch.stack(all_labels)
    
    # Compute positive and negative counts
    pos_count = all_labels.sum(0).float()
    neg_count = len(all_labels) - pos_count
    
    # Compute weights (avoid division by zero)
    pos_weight = neg_count / pos_count.clamp(min=1.0)
    pos_weight = pos_weight.clamp(max=max_weight)
    
    return pos_weight