import glob
import os
import random
import torch
from torch.utils.data import Dataset
from data import get_data_extended, get_data_extended_rotations
from tqdm import tqdm
from layers import OrigamiLayer
from util import pointcloud_sampling2, pointcloud_sampling3

class OrigamiDataset(Dataset):
    def __init__(self, root_dir, device='cpu', num_subdivisions=4, num_random_points_pool=32, cache_path="./cache/OrigamiDatasetCache.pt"):
        self.root_dir = root_dir
        self.device = device
        self.json_files = sorted(glob.glob(root_dir+"/**/*.json", recursive=True))
        
        if not self.json_files:
            raise ValueError(f"No json files found in {root_dir}")

        first_file = self.json_files[0]
        self.points, self.lines, self.faces, _, self.faces_indices_unique = get_data_extended(first_file)
        
        self.points = self.points.to(device)
        self.lines = self.lines.to(device)
        self.faces = self.faces.to(device)
        self.faces_indices_unique = self.faces_indices_unique.to(device)

        # Initialize solver for Ground Truth generation
        points_expanded = self.points.unsqueeze(0) 
        self.solver = OrigamiLayer(points_expanded, self.lines, self.faces).to(device)

        self.points_gt_sampling_list = []
        self.points_encode_sampling_pool_list = []
        self.target_theta_list = []

        if os.path.exists(cache_path):
            cache = torch.load(cache_path)
            self.points_gt_sampling_list = cache['points_gt_sampling_list']
            self.points_encode_sampling_pool_list = cache['points_encode_sampling_pool_list']
            self.target_theta_list = cache['target_theta_list']
            if 'json_files' in cache:
                self.json_files = cache['json_files']
            print("Loaded pre-computed ground truth and sampling from cache.")
            print(f"Loaded {len(self.json_files)} samples.")
        else:
            print("Pre-computing ground truth and sampling...")
            for json_file in tqdm(self.json_files):
                _, _, _, target_theta, _ = get_data_extended(json_file)
                target_theta = target_theta.to(device).unsqueeze(0) # (1, F, 1)

                with torch.no_grad():
                    points_gt = self.solver(target_theta, points_expanded)
                
                # Pre-sampling
                points_gt_sampling = pointcloud_sampling2(points_gt, self.faces_indices_unique, num_subdivisions=num_subdivisions)
                
                # Encode Sampling Pool (For Input)
                points_encode_sampling_pool = pointcloud_sampling3(points_gt, self.faces_indices_unique, num_random_points=num_random_points_pool)

                self.points_gt_sampling_list.append(points_gt_sampling.cpu())
                self.points_encode_sampling_pool_list.append(points_encode_sampling_pool.cpu())
                self.target_theta_list.append(target_theta.cpu())
            
            if not os.path.exists(os.path.dirname(cache_path)):
                os.makedirs(os.path.dirname(cache_path))
            torch.save({
                'json_files': self.json_files,
                'points_gt_sampling_list': self.points_gt_sampling_list,
                'points_encode_sampling_pool_list': self.points_encode_sampling_pool_list,
                'target_theta_list': self.target_theta_list
            }, cache_path)
            print(f"Loaded {len(self.json_files)} samples.")
            print(f"Saved pre-computed ground truth and sampling to {cache_path}")

    def __len__(self):
        return len(self.json_files)*2

    def __getitem__(self, idx):
        idx = idx % len(self.json_files)
        # Returns: input_points, points_gt_sampling
        
        pool = self.points_encode_sampling_pool_list[idx] # (1, M, 3)
        points_gt_sampling = self.points_gt_sampling_list[idx] # (1, M_gt, 3)
        
        num_points_total = pool.shape[1]
        indices = torch.randperm(num_points_total)[:256]
        input_points = pool[:, indices, :]
        
        return input_points.squeeze(0), points_gt_sampling.squeeze(0), self.target_theta_list[idx].squeeze(0)

class PhysicEngineDataset(Dataset):
    def __init__(self, root_dir:str, device:str='cpu', cache_path:str="./cache/PhysicEngineDatasetCache.pt", shuffle_seed:int|None=None):
        self.root_dir = root_dir
        self.device = device
        self.json_files = sorted(glob.glob(root_dir+"/**/*.json", recursive=True))
        
        if not self.json_files:
            raise ValueError(f"No json files found in {root_dir}")

        first_file = self.json_files[0]
        self.points, self.lines, self.faces, _, self.faces_indices_unique = get_data_extended(first_file)
        
        self.points = self.points.to(device)
        self.lines = self.lines.to(device)
        self.faces = self.faces.to(device)
        self.faces_indices_unique = self.faces_indices_unique.to(device)

        # Initialize solver for Ground Truth generation
        points_expanded = self.points.unsqueeze(0) 
        self.solver = OrigamiLayer(points_expanded, self.lines, self.faces).to(device)

        self.points_gt_list = []
        self.target_theta_list = []

        self.percents = [float(p) / 100.0 for p in range(-100, 101, 25)]
        self.num_scales = len(self.percents)  
        self.num_rotations = 4
        
        self.block_size = self.num_rotations * self.num_scales
        if os.path.exists(cache_path):
            cache = torch.load(cache_path)
            self.points_gt_list = cache['points_gt_list']
            self.target_theta_list = cache['target_theta_list']
            if 'json_files' in cache:
                self.json_files = cache['json_files']
            print("Loaded pre-computed ground truth.")
            print(f"Loaded {len(self.json_files)} samples.")
        else:
            print("Pre-computing ground truth...")
            scales = torch.tensor(self.percents, dtype=torch.float32).to(device) 
            
            for json_file in tqdm(self.json_files):
                sample_list = get_data_extended_rotations(json_file)
                all_thetas = []
                for sample in sample_list:
                    _, _, _, target_theta, _ = sample
                    target_theta = target_theta.to(device) 
                    rotation_thetas = target_theta.unsqueeze(0) * scales.view(self.num_scales, 1, 1)
                    all_thetas.append(rotation_thetas)
                batch_theta = torch.cat(all_thetas, dim=0)
                self.block_size = self.num_rotations * self.num_scales
                batch_points = points_expanded.expand(self.block_size, -1, -1)

                with torch.no_grad():
                    batch_points_gt = self.solver(batch_theta, batch_points)

                for i in range(self.block_size):
                    self.target_theta_list.append(batch_theta[i].squeeze(-1).cpu())
                    self.points_gt_list.append(batch_points_gt[i].cpu())

            if not os.path.exists(os.path.dirname(cache_path)):
                os.makedirs(os.path.dirname(cache_path))
            torch.save({
                'json_files': self.json_files,
                'points_gt_list': self.points_gt_list,
                'target_theta_list': self.target_theta_list
            }, cache_path)
            print(f"Loaded {len(self.json_files)} samples.")
            print(f"Saved pre-computed ground truth to {cache_path}")
        
        if shuffle_seed is not None:
            blocks_a = [self.points_gt_list[i:i+block_size] for i in range(0, len(self.points_gt_list), block_size)]
            blocks_b = [self.target_theta_list[i:i+block_size] for i in range(0, len(self.target_theta_list), block_size)]

            combined = list(zip(blocks_a, blocks_b, self.json_files))
            random.Random(shuffle_seed).shuffle(combined)
            blocks_a, blocks_b, self.json_files = zip(*combined)

            self.points_gt_list = [item for sublist in blocks_a for item in sublist]
            self.target_theta_list = [item for sublist in blocks_b for item in sublist]

    def __len__(self):
        return len(self.target_theta_list)

    def __getitem__(self, idx):
        num_rotate = (idx % self.block_size) // self.num_scales
        percentage = self.percents[(idx % self.block_size) % self.num_scales]

        return self.points_gt_list[idx], self.target_theta_list[idx], self.json_files[idx//self.block_size], num_rotate, percentage
        


if __name__ == "__main__":
    dataset = PhysicEngineDataset('./data/output_new_filtered')
    print(dataset[0][0].shape, dataset[0][1].shape)
    print(len(dataset))