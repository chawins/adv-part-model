import h5py
import torch
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    def __init__(self, root, mode, input_dim, transform=None, **kwargs):
        self.root = root
        self.mode = mode
        self.input_dim = input_dim
        self.transform = transform
        self.db = None

    def __len__(self):
        with h5py.File(self.root, "r") as db:
            return len(db.get(self.mode).get("images"))

    def __getitem__(self, idx: int):
        if self.db is None:
            self.db = h5py.File(self.root, "r").get(self.mode)

        images = torch.from_numpy(self.db.get("images")[idx])
        labels = self.db.get("labels")[idx]
        # part_indices = torch.from_numpy(self.db.get('part_indices')[idx])
        images = images.permute(0, 3, 1, 2)

        if self.transform is not None:
            new_images = torch.empty((images.size(0),) + self.input_dim)
            for i, image in enumerate(images):
                new_images[i] = self.transform(image)
            images = new_images

        output_size = list(self.input_dim)
        output_size[0] = images.size(0) * images.size(1)
        # return images.view(output_size), labels, part_indices
        return images.view(output_size), labels
