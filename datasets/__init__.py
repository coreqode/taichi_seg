from .colmap import ColmapDataset
from .nerf import NeRFDataset
from .nsvf import NSVFDataset
from .ngp import NGPDataset
from .brics import BRICSNGPDataset

dataset_dict = {
    'nerf': NeRFDataset,
    'nsvf': NSVFDataset,
    'colmap': ColmapDataset,
    'ngp': NGPDataset,
    'brics': BRICSNGPDataset,
}
