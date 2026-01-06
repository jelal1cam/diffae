# verify_alignment.py
"""
Diagnostic script to verify image-label alignment in CelebALMDBDataset.
Run this on the server to check if AUROC ~0.5 is due to label misalignment.
"""

import os
import sys
sys.path.insert(0, os.path.expanduser('~/diffae'))

import torch
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

from dataset import CelebALMDBDataset, BaseLMDB
from ro_optimization.evaluation.classification.data_utils import load_celeba_splits
from ro_optimization.evaluation.classification.config import get_config


def verify_lmdb_key_format():
    """Check what keys are actually in the LMDB."""
    print("=" * 60)
    print("1. Checking LMDB key format")
    print("=" * 60)

    cfg = get_config()
    lmdb_path = cfg.data.celeba.lmdb_file

    import lmdb
    env = lmdb.open(lmdb_path, readonly=True, lock=False)

    with env.begin() as txn:
        length = int(txn.get(b'length').decode('utf-8'))
        print(f"LMDB length: {length}")

        # Get first 5 keys
        cursor = txn.cursor()
        cursor.first()
        print("\nFirst 5 keys in LMDB:")
        for i, (key, _) in enumerate(cursor):
            if i >= 5:
                break
            print(f"  Key {i}: {key.decode('utf-8')}")

        # Get last key
        cursor.last()
        print(f"\nLast key: {cursor.key().decode('utf-8')}")

    env.close()


def verify_attribute_file_format():
    """Check attribute file structure."""
    print("\n" + "=" * 60)
    print("2. Checking attribute file format")
    print("=" * 60)

    cfg = get_config()
    attr_path = cfg.data.celeba.attr_file

    with open(attr_path) as f:
        line1 = f.readline().strip()
        line2 = f.readline().strip()
        line3 = f.readline().strip()
        line4 = f.readline().strip()

    print(f"Line 1 (count): {line1}")
    print(f"Line 2 (header): {line2[:100]}...")
    print(f"Line 3 (first data): {line3[:100]}...")
    print(f"Line 4 (second data): {line4[:100]}...")


def verify_partition_file():
    """Check partition file structure."""
    print("\n" + "=" * 60)
    print("3. Checking partition file")
    print("=" * 60)

    cfg = get_config()
    splits = load_celeba_splits(cfg.data.celeba.partition_file)

    print(f"Train images: {len(splits['train'])}")
    print(f"Val images: {len(splits['val'])}")
    print(f"Test images: {len(splits['test'])}")
    print(f"\nFirst 5 train images: {splits['train'][:5]}")
    print(f"First 5 val images: {splits['val'][:5]}")


def verify_dataset_indexing():
    """Check that dataset returns matching images and labels."""
    print("\n" + "=" * 60)
    print("4. Checking dataset indexing")
    print("=" * 60)

    cfg = get_config()
    splits = load_celeba_splits(cfg.data.celeba.partition_file)

    dataset = CelebALMDBDataset(
        lmdb_path=cfg.data.celeba.lmdb_file,
        attr_path=cfg.data.celeba.attr_file,
        image_size=128,
        do_augment=False,
        do_normalize=False,  # Keep in [0,1] for visualisation
        is_celebahq=False,
        split_files=splits['train']
    )

    print(f"Dataset length: {len(dataset)}")
    print(f"DataFrame shape: {dataset.df.shape}")
    print(f"\nFirst 5 rows of filtered DataFrame (index = filename):")
    print(dataset.df.head())

    # Check first 3 samples
    print("\n" + "-" * 40)
    print("Checking first 3 samples:")
    for i in range(3):
        sample = dataset[i]
        row = dataset.df.iloc[dataset.valid_indices[i]]

        print(f"\nSample {i}:")
        print(f"  DataFrame row index (filename): {row.name}")
        print(f"  Expected LMDB index: {int(row.name.split('.')[0]) - 1}")
        print(f"  Labels (first 5 attrs): {sample['labels'][:5].tolist()}")

        # Key attributes for easy visual verification
        male_idx = 20
        smiling_idx = 31
        eyeglasses_idx = 15
        print(f"  Male: {sample['labels'][male_idx]:.0f}, Smiling: {sample['labels'][smiling_idx]:.0f}, Eyeglasses: {sample['labels'][eyeglasses_idx]:.0f}")


def save_visual_samples():
    """Save sample images with labels for manual verification."""
    print("\n" + "=" * 60)
    print("5. Saving visual samples for manual verification")
    print("=" * 60)

    cfg = get_config()
    splits = load_celeba_splits(cfg.data.celeba.partition_file)

    dataset = CelebALMDBDataset(
        lmdb_path=cfg.data.celeba.lmdb_file,
        attr_path=cfg.data.celeba.attr_file,
        image_size=128,
        do_augment=False,
        do_normalize=False,  # Keep in [0,1] for visualisation
        is_celebahq=False,
        split_files=splits['train']
    )

    # Create output directory
    out_dir = "checkpoints/attribute_classifier/verification_samples"
    os.makedirs(out_dir, exist_ok=True)

    # Save first 10 samples
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, ax in enumerate(axes.flat):
        sample = dataset[i]
        img = sample['img'].permute(1, 2, 0).numpy()  # CHW -> HWC
        labels = sample['labels']

        row = dataset.df.iloc[dataset.valid_indices[i]]

        # Key attributes
        male = "M" if labels[20] > 0.5 else "F"
        smile = "Smile" if labels[31] > 0.5 else "No Smile"
        glasses = "Glasses" if labels[15] > 0.5 else "No Glasses"

        ax.imshow(img)
        ax.set_title(f"{row.name}\n{male}, {smile}, {glasses}", fontsize=8)
        ax.axis('off')

    plt.tight_layout()
    out_path = os.path.join(out_dir, "sample_verification.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved verification image to: {out_path}")
    print("Manually check if Male/Female, Smile/NoSmile, Glasses/NoGlasses labels match the faces!")


def check_label_distribution():
    """Check if labels are being loaded correctly (not all zeros or all ones)."""
    print("\n" + "=" * 60)
    print("6. Checking label distribution")
    print("=" * 60)

    cfg = get_config()
    splits = load_celeba_splits(cfg.data.celeba.partition_file)

    dataset = CelebALMDBDataset(
        lmdb_path=cfg.data.celeba.lmdb_file,
        attr_path=cfg.data.celeba.attr_file,
        image_size=128,
        do_augment=False,
        do_normalize=True,
        is_celebahq=False,
        split_files=splits['train']
    )

    # Sample 1000 random labels
    n_samples = min(1000, len(dataset))
    all_labels = []
    for i in range(n_samples):
        sample = dataset[i]
        all_labels.append(sample['labels'])

    all_labels = torch.stack(all_labels)

    print(f"Sampled {n_samples} labels")
    print(f"Label tensor shape: {all_labels.shape}")
    print(f"Mean per attribute (should be between 0.1-0.9 for most):")

    attr_names = [
        '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
        'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
        'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
        'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
        'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
        'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
        'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
        'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
        'Wearing_Necklace', 'Wearing_Necktie', 'Young'
    ]

    means = all_labels.mean(dim=0)
    for i, (name, mean) in enumerate(zip(attr_names, means)):
        flag = "OK" if 0.05 < mean < 0.95 else "EXTREME"
        print(f"  [{i:2d}] {name:25s}: {mean:.3f} {flag}")


if __name__ == "__main__":
    print("CelebA Dataset Alignment Verification")
    print("=" * 60)

    verify_lmdb_key_format()
    verify_attribute_file_format()
    verify_partition_file()
    verify_dataset_indexing()
    check_label_distribution()

    try:
        save_visual_samples()
    except Exception as e:
        print(f"Could not save visual samples (no display?): {e}")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    print("\nIf AUROC is still ~0.5 after running this, check:")
    print("1. Do the LMDB keys match the expected format?")
    print("2. Do the label distributions look reasonable?")
    print("3. Do the visual samples match their labels?")
