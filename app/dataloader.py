import argparse
import random
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from dataset import NuScenesTrajectoryDataset


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id: int) -> None:
	worker_seed = torch.initial_seed() % (2**32)
	np.random.seed(worker_seed)
	random.seed(worker_seed)


def _collate_batch(batch: Sequence[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    agent = torch.stack([item["agent"] for item in batch], dim=0)
    neighbors = torch.stack([item["neighbors"] for item in batch], dim=0)
    map_tensor = torch.stack([item["map"] for item in batch], dim=0)
    target = torch.stack([item["target"] for item in batch], dim=0)

    # 🔥 Create neighbor mask (1 = real, 0 = padded)
    neighbor_mask = (neighbors.abs().sum(dim=-1) != 0).float()
    # shape: [B, N, T_past]

    return {
        "agent": agent,
        "neighbors": neighbors,
        "neighbor_mask": neighbor_mask, 
        "map": map_tensor,
        "target": target,
    }


def create_dataset(
	dataroot: str,
	version: str,
	split: str,
	t_past: int,
	t_future: int,
	max_neighbors: int,
	neighbor_radius_m: float,
	map_size_meters: Tuple[float, float],
	map_canvas_size: Tuple[int, int],
	allowed_category_prefixes: Optional[Sequence[str]] = None,
) -> NuScenesTrajectoryDataset:
	return NuScenesTrajectoryDataset(
		dataroot=dataroot,
		version=version,
		split=split,
		t_past=t_past,
		t_future=t_future,
		max_neighbors=max_neighbors,
		neighbor_radius_m=neighbor_radius_m,
		map_size_meters=map_size_meters,
		map_canvas_size=map_canvas_size,
		allowed_category_prefixes=allowed_category_prefixes,
	)


def create_dataloader(
	dataset: NuScenesTrajectoryDataset,
	batch_size: int = 16,
	shuffle: bool = True,
	num_workers: int = 4,
	pin_memory: bool = True,
	drop_last: bool = False,
	persistent_workers: bool = True,
	seed: int = 42,
) -> DataLoader:
	generator = torch.Generator()
	generator.manual_seed(seed)

	if num_workers > 0:
		p_factor = 2              # Pre-load 2 batches per worker
		p_workers = persistent_workers
	else:
		p_factor = None           # Must be None if num_workers is 0
		p_workers = False         # Cannot be True if num_workers is 0

	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=shuffle,
		num_workers=num_workers,
		pin_memory=pin_memory,
		drop_last=drop_last,
		persistent_workers=persistent_workers,
		prefetch_factor=p_factor,
		collate_fn=_collate_batch,
		worker_init_fn=_seed_worker,
		generator=generator,
	)


def create_train_val_dataloaders(
	dataroot: str,
	version: str = "v1.0-mini",
	train_split: str = "train",
	val_split: str = "val",
	t_past: int = 4,
	t_future: int = 6,
	max_neighbors: int = 10,
	neighbor_radius_m: float = 20.0,
	map_size_meters: Tuple[float, float] = (20.0, 20.0),
	map_canvas_size: Tuple[int, int] = (224, 224),
	allowed_category_prefixes: Optional[Sequence[str]] = None,
	batch_size: int = 16,
	num_workers: int = 0,
	pin_memory: bool = True,
	persistent_workers: bool = False,
	seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
	set_seed(seed)

	train_dataset = create_dataset(
		dataroot=dataroot,
		version=version,
		split=train_split,
		t_past=t_past,
		t_future=t_future,
		max_neighbors=max_neighbors,
		neighbor_radius_m=neighbor_radius_m,
		map_size_meters=map_size_meters,
		map_canvas_size=map_canvas_size,
		allowed_category_prefixes=allowed_category_prefixes,
	)

	val_dataset = create_dataset(
		dataroot=dataroot,
		version=version,
		split=val_split,
		t_past=t_past,
		t_future=t_future,
		max_neighbors=max_neighbors,
		neighbor_radius_m=neighbor_radius_m,
		map_size_meters=map_size_meters,
		map_canvas_size=map_canvas_size,
		allowed_category_prefixes=allowed_category_prefixes,
	)

	train_loader = create_dataloader(
		dataset=train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,
		pin_memory=pin_memory,
		drop_last=True,
		persistent_workers=persistent_workers,
		seed=seed,
	)

	val_loader = create_dataloader(
		dataset=val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory,
		drop_last=False,
		persistent_workers=persistent_workers,
		seed=seed + 1,
	)

	return train_loader, val_loader


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="nuScenes dataloader sanity check")
	parser.add_argument("--dataroot", required=True, help="Path to nuScenes root directory")
	parser.add_argument("--version", default="v1.0-mini", help="nuScenes version folder name")
	parser.add_argument("--train_split", default="train", help="Train split name")
	parser.add_argument("--val_split", default="val", help="Validation split name")
	parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
	parser.add_argument("--num_workers", type=int, default=0, help="Number of dataloader workers")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	parser.add_argument("--max_neighbors", type=int, default=10, help="Max neighbors per agent")
	parser.add_argument("--neighbor_radius_m", type=float, default=20.0, help="Neighbor radius in meters")
	args = parser.parse_args()

	train_loader, val_loader = create_train_val_dataloaders(
		dataroot=args.dataroot,
		version=args.version,
		train_split=args.train_split,
		val_split=args.val_split,
		max_neighbors=args.max_neighbors,
		neighbor_radius_m=args.neighbor_radius_m,
		batch_size=args.batch_size,
		num_workers=args.num_workers,
		seed=args.seed,
	)

	print(f"Train samples: {len(train_loader.dataset)}")
	print(f"Val samples: {len(val_loader.dataset)}")

	train_batch = next(iter(train_loader))
	print("Train batch shapes:")
	print(f"  agent: {tuple(train_batch['agent'].shape)}")
	print(f"  neighbors: {tuple(train_batch['neighbors'].shape)}")
	print(f"  map: {tuple(train_batch['map'].shape)}")
	print(f"  target: {tuple(train_batch['target'].shape)}")

	val_batch = next(iter(val_loader))
	print("Val batch shapes:")
	print(f"  agent: {tuple(val_batch['agent'].shape)}")
	print(f"  neighbors: {tuple(val_batch['neighbors'].shape)}")
	print(f"  map: {tuple(val_batch['map'].shape)}")
	print(f"  target: {tuple(val_batch['target'].shape)}")
	print(val_batch["agent"].abs().mean())
	print(val_batch["neighbors"].abs().mean())
