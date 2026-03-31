import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from pyquaternion import Quaternion
from scipy.spatial import KDTree
from torch import Tensor
from torch.utils.data import Dataset

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes


class NuScenesTrajectoryDataset(Dataset):
	"""nuScenes dataset for intent-aware trajectory prediction.

	Each item returns:
	- agent: Tensor [T_past, 4] -> [x, y, vx, vy] in agent-centric coordinates
	- neighbors: Tensor [N, 4] -> [dx, dy, dvx, dvy] (zero-padded to max_neighbors)
	- map: Tensor [3, H, W] -> local map crop centered on the agent
	- target: Tensor [T_future, 2] -> [x, y] in agent-centric coordinates
	"""

	def __init__(
		self,
		dataroot: str,
		version: str = "v1.0-mini",
		split: str = "train",
		t_past: int = 4,
		t_future: int = 6,
		max_neighbors: int = 10,
		neighbor_radius_m: float = 20.0,
		map_size_meters: Tuple[float, float] = (20.0, 20.0),
		map_canvas_size: Tuple[int, int] = (224, 224),
		allowed_category_prefixes: Optional[Sequence[str]] = None,
	) -> None:
		self.dataroot = str(Path(dataroot).expanduser().resolve())
		self.version = self._resolve_version(self.dataroot, version)
		self.split = self._resolve_split(split, self.version)
		self.t_past = t_past
		self.t_future = t_future
		self.max_neighbors = max_neighbors
		self.neighbor_radius_m = neighbor_radius_m
		self.map_size_meters = map_size_meters
		self.map_canvas_size = map_canvas_size
		self.allowed_category_prefixes = tuple(
			allowed_category_prefixes
			if allowed_category_prefixes is not None
			else ("human.pedestrian", "vehicle.bicycle", "vehicle.motorcycle")
		)

		available_versions = self._detect_available_versions(self.dataroot)
		try:
			self.nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=False)
		except Exception as exc:
			available_text = ", ".join(available_versions) if available_versions else "none detected"
			raise ValueError(
				"Failed to initialize nuScenes. "
				f"Requested version='{self.version}', dataroot='{self.dataroot}'. "
				f"Available versions under dataroot: {available_text}. "
				"Ensure the dataset is extracted as dataroot/<version>/... "
				"(for example dataroot/v1.0-mini/sample.json)."
			) from exc

		# Lightweight caches to reduce repeated devkit table lookups.
		self._ann_cache: Dict[str, Dict] = {}
		self._sample_cache: Dict[str, Dict] = {}
		self._scene_cache: Dict[str, Dict] = {}
		self._log_cache: Dict[str, Dict] = {}
		self._velocity_cache: Dict[str, np.ndarray] = {}
		self._sample_instance_to_ann: Dict[str, Dict[str, str]] = {}
		self._sample_allowed_ann_tokens: Dict[str, List[str]] = {}
		self._sample_location_cache: Dict[str, str] = {}
		self._map_cache: Dict[str, Optional[NuScenesMap]] = {}

		self._items: List[str] = self._build_index()
		if len(self._items) == 0:
			raise ValueError(
				"No valid samples found after indexing. "
				f"Resolved split='{self.split}', version='{self.version}'. "
				"Check dataroot contents and split/version compatibility."
			)

		# Per-item context set in __getitem__, used by helper signatures requested by user.
		self._current_instance_token: Optional[str] = None
		self._current_agent_velocity = np.zeros((2,), dtype=np.float32)

	def _get_agent_yaw(self, ann_token: str) -> float:
		ann = self._get_ann(ann_token)
		return Quaternion(ann["rotation"]).yaw_pitch_roll[0]

	def _transform_to_agent_frame(self, x, y, agent_x, agent_y, agent_yaw):
		dx = x - agent_x
		dy = y - agent_y

		cos_yaw = np.cos(agent_yaw)
		sin_yaw = np.sin(agent_yaw)

		x_local = cos_yaw * dx + sin_yaw * dy
		y_local = -sin_yaw * dx + cos_yaw * dy

		return x_local, y_local

	def _rotate_velocity(self, vx, vy, agent_yaw):
		cos_yaw = np.cos(agent_yaw)
		sin_yaw = np.sin(agent_yaw)

		vx_local = cos_yaw * vx + sin_yaw * vy
		vy_local = -sin_yaw * vx + cos_yaw * vy

		return vx_local, vy_local

	@staticmethod
	def _detect_available_versions(dataroot: str) -> List[str]:
		root = Path(dataroot)
		if not root.exists() or not root.is_dir():
			return []

		versions: List[str] = []
		for child in root.iterdir():
			if child.is_dir() and child.name.startswith("v1.0-") and (child / "sample.json").exists():
				versions.append(child.name)
		return sorted(versions)

	@classmethod
	def _resolve_version(cls, dataroot: str, requested_version: str) -> str:
		available = cls._detect_available_versions(dataroot)
		if requested_version in available:
			return requested_version
		if len(available) == 1:
			return available[0]
		return requested_version

	@staticmethod
	def _resolve_split(split: str, version: str) -> str:
		if version == "v1.0-mini":
			if split == "train":
				return "mini_train"
			if split == "val":
				return "mini_val"
		else:
			if split == "mini_train":
				return "train"
			if split == "mini_val":
				return "val"
		return split

	def _is_allowed_agent(self, ann: Dict) -> bool:
		category = ann.get("category_name", "")
		return any(category.startswith(prefix) for prefix in self.allowed_category_prefixes)

	def _get_ann(self, ann_token: str) -> Dict:
		if ann_token not in self._ann_cache:
			self._ann_cache[ann_token] = self.nusc.get("sample_annotation", ann_token)
		return self._ann_cache[ann_token]

	def _get_sample(self, sample_token: str) -> Dict:
		if sample_token not in self._sample_cache:
			self._sample_cache[sample_token] = self.nusc.get("sample", sample_token)
		return self._sample_cache[sample_token]

	def _get_scene(self, scene_token: str) -> Dict:
		if scene_token not in self._scene_cache:
			self._scene_cache[scene_token] = self.nusc.get("scene", scene_token)
		return self._scene_cache[scene_token]

	def _get_log(self, log_token: str) -> Dict:
		if log_token not in self._log_cache:
			self._log_cache[log_token] = self.nusc.get("log", log_token)
		return self._log_cache[log_token]

	def _get_velocity_xy(self, ann_token: str) -> np.ndarray:
		if ann_token in self._velocity_cache:
			return self._velocity_cache[ann_token]

		vel_xyz = self.nusc.box_velocity(ann_token)
		if vel_xyz is None or np.any(np.isnan(vel_xyz)):
			ann = self._get_ann(ann_token)
			if ann["prev"] != "":
				prev_ann = self._get_ann(ann["prev"])
				t0 = float(self._get_sample(prev_ann["sample_token"])["timestamp"]) / 1_000_000.0
				t1 = float(self._get_sample(ann["sample_token"])["timestamp"]) / 1_000_000.0
				dt = max(t1 - t0, 1e-3)
				vx = (float(ann["translation"][0]) - float(prev_ann["translation"][0])) / dt
				vy = (float(ann["translation"][1]) - float(prev_ann["translation"][1])) / dt
				vel = np.asarray([vx, vy], dtype=np.float32)
			else:
				vel = np.zeros((2,), dtype=np.float32)
		else:
			vel = np.asarray([vel_xyz[0], vel_xyz[1]], dtype=np.float32)

		self._velocity_cache[ann_token] = vel
		return vel

	def _build_index(self) -> List[str]:
		split_scenes = set(create_splits_scenes().get(self.split, []))
		if not split_scenes:
			raise ValueError(f"Unknown split '{self.split}'. Use one of: train, val, test, mini_train, mini_val.")

		valid_scene_tokens = {scene["token"] for scene in self.nusc.scene if scene["name"] in split_scenes}
		items: List[str] = []
		for sample in self.nusc.sample:
			if sample["scene_token"] not in valid_scene_tokens:
				continue

			sample_token = sample["token"]
			instance_to_ann: Dict[str, str] = {}
			allowed_ann_tokens: List[str] = []
			for ann_token in sample["anns"]:
				ann = self._get_ann(ann_token)
				instance_to_ann[ann["instance_token"]] = ann_token
				if self._is_allowed_agent(ann):
					allowed_ann_tokens.append(ann_token)

			self._sample_instance_to_ann[sample_token] = instance_to_ann
			self._sample_allowed_ann_tokens[sample_token] = allowed_ann_tokens

			for ann_token in allowed_ann_tokens:
				if self._has_required_context(ann_token):
					items.append(ann_token)
		return items

	def _has_required_context(self, ann_token: str) -> bool:
		cur_tok = ann_token
		for _ in range(self.t_past - 1):
			cur_tok = self._get_ann(cur_tok)["prev"]
			if cur_tok == "":
				return False

		cur_tok = ann_token
		for _ in range(self.t_future):
			cur_tok = self._get_ann(cur_tok)["next"]
			if cur_tok == "":
				return False
		return True

	def __len__(self) -> int:
		return len(self._items)

	def __getitem__(self, idx: int) -> Dict[str, Tensor]:
		ann_token = self._items[idx]
		ann = self._get_ann(ann_token)
		sample_token = ann["sample_token"]
		instance_token = ann["instance_token"]

		self._current_instance_token = instance_token
		agent_past, target_future = self._get_agent_trajectory(sample_token, instance_token)
		agent_x, agent_y = float(ann["translation"][0]), float(ann["translation"][1])
		neighbors = self._get_neighbors(sample_token, agent_x, agent_y)
		map_tensor = self._get_map_crop(sample_token, agent_x, agent_y)

		return {
			"agent": agent_past,
			"neighbors": neighbors,
			"map": map_tensor,
			"target": target_future,
		}


	def _get_agent_trajectory(self, sample_token, instance_token):
		current_ann_token = self._sample_instance_to_ann.get(sample_token, {}).get(instance_token, "")
		if current_ann_token == "":
			raise ValueError("Could not find target instance in provided sample.")

		current_ann = self._get_ann(current_ann_token)
		agent_x = float(current_ann["translation"][0])
		agent_y = float(current_ann["translation"][1])
		agent_yaw = self._get_agent_yaw(current_ann_token)

		# Collect tokens
		past_tokens = [current_ann_token]
		cur_tok = current_ann_token
		for _ in range(self.t_past - 1):
			prev_tok = self._get_ann(cur_tok)["prev"]
			if prev_tok == "":
				break
			past_tokens.append(prev_tok)
			cur_tok = prev_tok
		past_tokens.reverse()

		future_tokens = []
		cur_tok = current_ann_token
		for _ in range(self.t_future):
			next_tok = self._get_ann(cur_tok)["next"]
			if next_tok == "":
				break
			future_tokens.append(next_tok)
			cur_tok = next_tok

		agent_rows = []
		for tok in past_tokens:
			ann = self._get_ann(tok)

			x_global = float(ann["translation"][0])
			y_global = float(ann["translation"][1])

			x, y = self._transform_to_agent_frame(
				x_global, y_global, agent_x, agent_y, agent_yaw
			)

			vel = self._get_velocity_xy(tok)
			vx, vy = self._rotate_velocity(vel[0], vel[1], agent_yaw)

			agent_rows.append([x, y, vx, vy])

		future_rows = []
		for tok in future_tokens:
			ann = self._get_ann(tok)

			x_global = float(ann["translation"][0])
			y_global = float(ann["translation"][1])

			x, y = self._transform_to_agent_frame(
				x_global, y_global, agent_x, agent_y, agent_yaw
			)

			future_rows.append([x, y])

		agent_past = torch.zeros((self.t_past, 4), dtype=torch.float32)
		target_future = torch.zeros((self.t_future, 2), dtype=torch.float32)

		if agent_rows:
			agent_np = np.asarray(agent_rows[-self.t_past:], dtype=np.float32)
			agent_past[-agent_np.shape[0]:] = torch.from_numpy(agent_np)
			self._current_agent_velocity = agent_np[-1, 2:4].copy()
		else:
			self._current_agent_velocity = np.zeros((2,), dtype=np.float32)

		if future_rows:
			future_np = np.asarray(future_rows[:self.t_future], dtype=np.float32)
			target_future[:future_np.shape[0]] = torch.from_numpy(future_np)

		print("---- DEBUG ----")
		for tok in past_tokens:
			ann = self._get_ann(tok)
			print("Global:", ann["translation"][:2])

		return agent_past, target_future

	def _get_neighbors(self, sample_token, agent_x, agent_y):
		current_ann_token = self._sample_instance_to_ann[sample_token][self._current_instance_token]
		agent_yaw = self._get_agent_yaw(current_ann_token)

		positions = []
		features = []

		for tok in self._sample_allowed_ann_tokens.get(sample_token, []):
			ann = self._get_ann(tok)

			if ann["instance_token"] == self._current_instance_token:
				continue

			x = float(ann["translation"][0])
			y = float(ann["translation"][1])
			vel = self._get_velocity_xy(tok)

			positions.append([x, y])
			features.append([x, y, vel[0], vel[1]])

		neighbors = torch.zeros((self.max_neighbors, self.t_past, 4), dtype=torch.float32)

		if not positions:
			return neighbors
		pos_np = np.asarray(positions, dtype=np.float32)

		# KDTree caching
		if not hasattr(self, "_kdtree_cache"):
			self._kdtree_cache = {}

		if sample_token not in self._kdtree_cache:
			self._kdtree_cache[sample_token] = KDTree(pos_np)

		tree = self._kdtree_cache[sample_token]

		indices = tree.query_ball_point([agent_x, agent_y], r=self.neighbor_radius_m)

		if not indices:
			return neighbors

		selected = sorted(
			indices,
			key=lambda i: np.linalg.norm(pos_np[i] - np.array([agent_x, agent_y]))
		)[:self.max_neighbors]

		for i, idx in enumerate(selected):
			x, y, vx, vy = features[idx]

			x_local, y_local = self._transform_to_agent_frame(
				x, y, agent_x, agent_y, agent_yaw
			)

			vx_local, vy_local = self._rotate_velocity(vx, vy, agent_yaw)

			vx_local -= float(self._current_agent_velocity[0])
			vy_local -= float(self._current_agent_velocity[1])

			# Repeat across time for STGCN compatibility
			for t in range(self.t_past):
				neighbors[i, t] = torch.tensor(
					[x_local, y_local, vx_local, vy_local], dtype=torch.float32
				)

		return neighbors

	def _get_map(self, location: str) -> Optional[NuScenesMap]:
		if location not in self._map_cache:
			try:
				self._map_cache[location] = NuScenesMap(dataroot=self.dataroot, map_name=location)
			except Exception:
				self._map_cache[location] = None
		return self._map_cache[location]

	def _get_map_crop(self, sample_token, agent_x, agent_y):
		# 1. Fetch map for the current sample location.
		sample = self._get_sample(sample_token)
		location = self._sample_location_cache.get(sample_token, "")
		if location == "":
			scene = self._get_scene(sample["scene_token"])
			log = self._get_log(scene["log_token"])
			location = log["location"]
			self._sample_location_cache[sample_token] = location
		nusc_map = self._get_map(location)
		if nusc_map is None:
			h, w = self.map_canvas_size
			return torch.zeros((3, h, w), dtype=torch.float32)

		# 2. Crop map to 20m x 20m centered on the agent.
		patch_w, patch_h = self.map_size_meters
		patch_box = (float(agent_x), float(agent_y), float(patch_w), float(patch_h))

		# Use the current agent heading if available in this sample.
		agent_yaw = 0.0
		if self._current_instance_token is not None:
			agent_tok = self._sample_instance_to_ann.get(sample_token, {}).get(self._current_instance_token, "")
			if agent_tok != "":
				agent_yaw = self._get_agent_yaw(agent_tok)

		patch_angle_deg = agent_yaw * 180.0 / np.pi

		layer_names = ["drivable_area", "walkway", "ped_crossing"]
		try:
			map_mask = nusc_map.get_map_mask(
				patch_box=patch_box,
				patch_angle=patch_angle_deg,
				layer_names=layer_names,
				canvas_size=self.map_canvas_size,
			).astype(np.float32)
		except Exception:
			h, w = self.map_canvas_size
			return torch.zeros((3, h, w), dtype=torch.float32)

		# If a backend returns unexpected spatial size, resize each channel with OpenCV.
		h, w = self.map_canvas_size
		if map_mask.shape[1] != h or map_mask.shape[2] != w:
			resized = []
			for c in range(map_mask.shape[0]):
				resized.append(cv2.resize(map_mask[c], (w, h), interpolation=cv2.INTER_NEAREST))
			map_mask = np.stack(resized, axis=0).astype(np.float32)

		return torch.from_numpy(map_mask)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="nuScenes trajectory dataset sanity check")
	parser.add_argument("--dataroot", required=True, help="Path to nuScenes root directory")
	parser.add_argument("--version", default="v1.0-mini", help="nuScenes version folder name")
	parser.add_argument("--split", default="train", help="Split name (train/val/test or mini_train/mini_val)")
	args = parser.parse_args()

	dataset = NuScenesTrajectoryDataset(
		dataroot=args.dataroot,
		version=args.version,
		split=args.split,
	)

	print(f"Dataset size: {len(dataset)}")
	sample = dataset[0]
	agent = sample["agent"].numpy()
	target = sample["target"].numpy()
	neighbors = sample["neighbors"][:, 0, :].numpy()  # current timestep
	map_img = sample["map"].numpy().transpose(1, 2, 0)
	print("Agent trajectory (x, y, vx, vy):")
	print(agent)
	print("Target future positions (x, y):")
	print(target)	
	print("Neighbors (dx, dy, dvx, dvy):")
	print(neighbors)
	print("Map crop shape:", map_img.shape)
