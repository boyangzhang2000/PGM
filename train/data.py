"""Recursive RGB datasets with deterministic crops and epoch permutations."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, Sampler


class ImageDataset(Dataset):
    def __init__(self, root, image_size, recursive=True, crop='center', horizontal_flip=False, seed=0):
        self.root = Path(root).resolve()
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        candidates = self.root.rglob('*') if recursive else self.root.glob('*')
        self.paths = sorted(p for p in candidates if p.is_file() and p.suffix.lower() in extensions)
        if not self.paths:
            raise FileNotFoundError(f"No training images: {self.root}")
        self.image_size, self.crop = image_size, crop
        self.horizontal_flip, self.seed = horizontal_flip, seed

    def __len__(self):
        return len(self.paths)

    def signature(self):
        entries = [(p.relative_to(self.root).as_posix(), p.stat().st_size, p.stat().st_mtime_ns)
                   for p in self.paths]
        return hashlib.sha256(json.dumps(entries, ensure_ascii=False).encode()).hexdigest()

    def __getitem__(self, key):
        epoch, index = key if isinstance(key, tuple) else (0, key)
        generator = torch.Generator().manual_seed(self.seed + epoch*len(self) + index)
        try:
            with Image.open(self.paths[index]) as source:
                im = ImageOps.exif_transpose(source).convert('RGB')
                width, height = im.size
                scale = max(1., self.image_size/min(width, height))
                if scale > 1:
                    im = im.resize((math.ceil(width*scale), math.ceil(height*scale)), Image.Resampling.BICUBIC)
                width, height = im.size
                if self.crop == 'random':
                    left = int(torch.randint(width-self.image_size+1, (), generator=generator))
                    top = int(torch.randint(height-self.image_size+1, (), generator=generator))
                else:
                    left, top = (width-self.image_size)//2, (height-self.image_size)//2
                im = im.crop((left, top, left+self.image_size, top+self.image_size))
                if self.horizontal_flip and bool(torch.randint(2, (), generator=generator)):
                    im = ImageOps.mirror(im)
                array = np.array(im, dtype=np.float32)/127.5-1
            return torch.from_numpy(array).permute(2, 0, 1).contiguous()
        except Exception as error:
            raise RuntimeError(f"Cannot read training image: {self.paths[index]}") from error


class EpochBatches(Sampler):
    def __init__(self, count, batch_size, seed, epoch, start_batch=0):
        self.count, self.batch_size, self.seed = count, batch_size, seed
        self.epoch, self.start_batch = epoch, start_batch

    def __len__(self):
        return math.ceil(self.count/self.batch_size)-self.start_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed+self.epoch)
        order = torch.randperm(self.count, generator=generator).tolist()
        for start in range(self.start_batch*self.batch_size, self.count, self.batch_size):
            yield [(self.epoch, index) for index in order[start:start+self.batch_size]]
