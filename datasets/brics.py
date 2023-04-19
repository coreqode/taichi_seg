import json
import os

import numpy as np
import torch
from tqdm import tqdm

from .base import BaseDataset
from .color_utils import read_image
from .ray_utils import get_ray_directions
from datasets.ray_utils import get_rays

class BRICSNGPDataset(BaseDataset):

    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        if kwargs.get('read_meta', True):
            self.read_meta(split)

    def read_intrinsics(self, meta):
        w = int(meta['w'] * self.downsample)
        h = int(meta['h'] * self.downsample)
        # fx = fy = 0.5 * 800 / np.tan(
        #     0.5 * meta['camera_angle_x']) * self.downsample
    
        # fx = 0.5 * w / np.tan(0.5 * meta['camera_angle_x']) * self.downsample
        # fy = 0.5 * h / np.tan(0.5 * meta['camera_angle_y']) * self.downsample
        fx = meta['fl_x'] * self.downsample
        fy = meta['fl_y'] * self.downsample

        K = np.float32([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1]]) 
        K = torch.FloatTensor(K)

        directions = get_ray_directions(h, w, torch.FloatTensor(K))
        return directions, K

    def read_meta(self, split):
        self.rays = []
        self.poses = []
        self.directions = []
        self.K = []

        with open(os.path.join(self.root_dir, "transforms.json"), 'r') as f:
            frames = json.load(f)

        self.img_wh = (int(frames[0]['w'] * self.downsample), int(frames[0]['h'] * self.downsample))
        print(f'Loading {len(frames)} {split} images ...')
        for frame in tqdm(frames):
            img_path = os.path.join(self.root_dir, f"{frame['file_path']}")
            directions, K = self.read_intrinsics(frame)
            self.directions.append(directions)
            self.K.append(K)
            if os.path.exists(img_path):
                c2w = np.array(frame['transform_matrix'])[:3, :4]

                self.poses += [c2w]
                try:
                    img = read_image(img_path, self.img_wh)
                    self.rays += [img]
                except OSError as e:
                    print("can not read image", e)

        if len(self.rays) > 0:
            self.rays = torch.FloatTensor(np.stack(
                self.rays))  # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses)  # (N_images, 3, 4)
        self.directions = torch.stack(self.directions)  # (N_images, hw, 3)
        self.K = torch.stack(self.K)
        
        self.rays_origin = [] 
        self.rays_direction = []
        for i in range(self.poses.shape[0]):
            rays_o, rays_d = get_rays(self.directions[i], self.poses[i])
            self.rays_origin.append(rays_o)
            self.rays_direction.append(rays_d)

        self.rays_origin = torch.stack(self.rays_origin)
        self.rays_direction = torch.stack(self.rays_direction)
