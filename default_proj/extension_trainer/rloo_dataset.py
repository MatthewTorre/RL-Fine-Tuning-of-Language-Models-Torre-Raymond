"""Minimal dataset wrapper for RLOO sampling.

Unlike SFT/IPO, this loader returns raw prompt strings and structured
ground-truth metadata because responses are generated online.
"""

from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import os
import random

class RLOODataset(Dataset):
    """Loads prompts plus per-example reward metadata."""
    def __init__(
        self,
        dataset_name,
        split='train',
        batch_size=16,
        num_proc=os.cpu_count(),
        fraction=None,
        max_examples=None,
        subset_seed=0,
        subset_strategy='first',
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.batch_size = batch_size
        self.num_proc = num_proc

        self.dataset = load_dataset(dataset_name, split=split, num_proc=num_proc)
        self.original_size = len(self.dataset)
        self.selected_indices = None
        self._apply_subset(
            fraction=fraction,
            max_examples=max_examples,
            subset_seed=subset_seed,
            subset_strategy=subset_strategy,
        )
        self.all_prompts = self.dataset['prompt']
        self.all_ground_truth = self.dataset['ground_truth']

    def _apply_subset(self, fraction=None, max_examples=None, subset_seed=0, subset_strategy='first'):
        """Optionally shrink the loaded split to a deterministic index pool."""
        subset_size = self.original_size
        if fraction is not None:
            if not 0.0 < fraction <= 1.0:
                raise ValueError(f"fraction must be in (0, 1], got {fraction}")
            subset_size = min(subset_size, max(1, int(self.original_size * fraction)))
        if max_examples is not None:
            if max_examples <= 0:
                raise ValueError(f"max_examples must be positive, got {max_examples}")
            subset_size = min(subset_size, max_examples)

        if subset_size == self.original_size:
            return

        if subset_strategy == 'first':
            selected_indices = list(range(subset_size))
        elif subset_strategy == 'random':
            rng = random.Random(subset_seed)
            selected_indices = rng.sample(range(self.original_size), subset_size)
        else:
            raise ValueError(
                f"subset_strategy must be one of ('first', 'random'), got {subset_strategy}"
            )

        self.selected_indices = selected_indices
        self.dataset = self.dataset.select(selected_indices)

    def __len__(self):
        return len(self.all_prompts)

    def __getitem__(self, idx):
        prompt = self.all_prompts[idx]
        ground_truth = self.all_ground_truth[idx]
        return {'id': idx, 'prompt': prompt, 'ground_truth': ground_truth}

    def collate_fn(self, batch):
        # Keep items as Python objects; tokenization happens after sampling.
        ids = [item['id'] for item in batch]
        prompts = [item['prompt'] for item in batch]
        ground_truths = [item['ground_truth'] for item in batch]
        return {'ids': ids, 'prompt': prompts, 'ground_truth': ground_truths}

    def batch_by_ids(self, ids):
        """Build a trainer batch for explicit dataset row IDs."""
        ids = [int(idx) for idx in ids]
        prompts = [self.all_prompts[idx] for idx in ids]
        ground_truths = [self.all_ground_truth[idx] for idx in ids]
        return {'ids': ids, 'prompt': prompts, 'ground_truth': ground_truths}


def get_dataloaders(
    dataset_name,
    splits=['train', 'test'],
    batch_size=16,
    num_proc=4,
    shuffle=True,
    train_fraction=None,
    train_max_examples=None,
    train_subset_seed=0,
    train_subset_strategy='first',
):
    """Create split->DataLoader dict for online RL sampling."""
    dataloaders = {}
    for split in splits:
        is_train_split = split == 'train'
        dataset = RLOODataset(
            dataset_name=dataset_name,
            split=split,
            batch_size=batch_size,
            num_proc=num_proc,
            fraction=train_fraction if is_train_split else None,
            max_examples=train_max_examples if is_train_split else None,
            subset_seed=train_subset_seed,
            subset_strategy=train_subset_strategy,
        )
        dataloaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            num_workers=num_proc,
            pin_memory=True,
            drop_last=True
        )
    return dataloaders
