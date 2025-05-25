from torch.utils.data import Dataset, DataLoader
import torch
import os
import json
import h5py
import random
from cadlib.macro import *


def get_dataloader(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = CADDataset(phase, config)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader


class CADDataset(Dataset):
    def __init__(self, phase, config):
        super(CADDataset, self).__init__()
        self.raw_data = os.path.join(config.data_root, "cad_vec") # h5 data root
        self.phase = phase
        self.aug = config.augment
        self.path = os.path.join(config.data_root, "train_val_test_split.json")
        with open(self.path, "r") as fp:
            self.all_data = json.load(fp)[phase]

        self.max_n_loops = config.max_n_loops          # Number of paths (N_P)
        self.max_n_curves = config.max_n_curves            # Number of commands (N_C)
        self.max_total_len = config.max_total_len
        self.size = 256

    def get_data_by_id(self, data_id):
        idx = self.all_data.index(data_id)
        return self.__getitem__(idx)

    def __getitem__(self, index):
        data_id = self.all_data[index]
        h5_path = os.path.join(self.raw_data, data_id + ".h5")
        with h5py.File(h5_path, "r") as fp:
            cad_vec = fp["vec"][:] # (len, 1 + N_ARGS)

        if self.aug and self.phase == "train":
            command1 = cad_vec[:, 0]
            ext_indices1 = np.where(command1 == EXT_IDX)[0]
            # if len(ext_indices1) > 1 and random.randint(0, 1) == 1:
            if len(ext_indices1) > 1 and random.uniform(0, 1) > 0.5:
                ext_vec1 = np.split(cad_vec, ext_indices1 + 1, axis=0)[:-1]
        
                data_id2 = self.all_data[random.randint(0, len(self.all_data) - 1)]
                h5_path2 = os.path.join(self.raw_data, data_id2 + ".h5")
                with h5py.File(h5_path2, "r") as fp:
                    cad_vec2 = fp["vec"][:]
                command2 = cad_vec2[:, 0]
                ext_indices2 = np.where(command2 == EXT_IDX)[0]
                ext_vec2 = np.split(cad_vec2, ext_indices2 + 1, axis=0)[:-1]
        
                n_replace = random.randint(1, min(len(ext_vec1) - 1, len(ext_vec2)))
                old_idx = sorted(random.sample(list(range(len(ext_vec1))), n_replace))
                new_idx = sorted(random.sample(list(range(len(ext_vec2))), n_replace))
                for i in range(len(old_idx)):
                    ext_vec1[old_idx[i]] = ext_vec2[new_idx[i]]
        
                sum_len = 0
                new_vec = []
                for i in range(len(ext_vec1)):
                    sum_len += len(ext_vec1[i])
                    if sum_len > self.max_total_len:
                        break
                    new_vec.append(ext_vec1[i])
                cad_vec = np.concatenate(new_vec, axis=0)

        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        command = cad_vec[:, 0]
        args = cad_vec[:, 1:]
        command = torch.tensor(command, dtype=torch.long)
        args = torch.tensor(args, dtype=torch.long)
        return {"command": command, "args": args, "id": data_id}

    def __len__(self):
        return len(self.all_data)

# Define MATERIAL_TABLE and number of materials
MATERIAL_TABLE = {"Al": 0, "Fe": 1, "Ni": 2, "Cr": 3, "Cu": 4, "Au": 5}
N_MATERIALS = len(MATERIAL_TABLE)


class CADDatasetWithCondition(Dataset):
    def __init__(self, phase, config, label_json_path):
        super().__init__()
        self.raw_data = os.path.join(config.data_root, "cad_vec")
        self.phase = phase
        self.aug = config.augment
        self.max_n_loops = config.max_n_loops
        self.max_n_curves = config.max_n_curves
        self.max_total_len = config.max_total_len
        self.size = 256

        # Load split
        split_path = os.path.join(config.data_root, "train_val_test_split.json")
        with open(split_path, "r") as f:
            all_split_ids = json.load(f)[phase]

        # Load and filter labels
        with open(label_json_path, "r") as f:
            raw_labels = json.load(f)

        self.label_dict = {
            k: v for k, v in raw_labels.items()
            if v.get("valid", True) and v.get("wall_thickness") is not None
        }

        # Filter to only data_ids with valid metadata
        self.all_data = [
            d for d in all_split_ids if os.path.basename(d) in self.label_dict
        ]

        # Precompute normalization stats
        vols = np.array([v["volume"] for v in self.label_dict.values()])
        masses = np.array([v["mass"] for v in self.label_dict.values()])
        thicks = np.array([v["wall_thickness"] for v in self.label_dict.values()])

        # self.vmin, self.vmax = vols.min(), vols.max()
        # self.mmin, self.mmax = masses.min(), masses.max()
        # self.tmin, self.tmax = thicks.min(), thicks.max()

    def normalize(self, x, xmin, xmax):
        return (x - xmin) / (xmax - xmin + 1e-8)

    def get_condition_vector(self, data_id):
        meta = self.label_dict[os.path.basename(data_id)]
        vol = meta["volume"]
        mass = meta["mass"]
        thick = meta["wall_thickness"]
        # vol = np.ceil(meta["volume"] * 1e4) / 1e4
        # mass = np.ceil(meta["mass"] * 1e4) / 1e4
        # # Round thickness down to nearest .0001
        # thick = np.floor(meta["wall_thickness"] * 1e4) / 1e4
        mat_id = MATERIAL_TABLE[meta["material"]]
        mat_onehot = np.eye(N_MATERIALS)[mat_id]
        return torch.tensor(np.concatenate([[vol, mass, thick], mat_onehot]), dtype=torch.float32)

    def __getitem__(self, index):
        data_id = self.all_data[index]
        h5_path = os.path.join(self.raw_data, data_id + ".h5")

        with h5py.File(h5_path, "r") as f:
            cad_vec = f["vec"][:]

        if self.aug and self.phase == "train":
            cad_vec = self._augment_vector(cad_vec)

        # Pad
        pad_len = self.max_total_len - cad_vec.shape[0]
        cad_vec = np.concatenate([cad_vec, EOS_VEC[np.newaxis].repeat(pad_len, axis=0)], axis=0)

        command = torch.tensor(cad_vec[:, 0], dtype=torch.long)
        args = torch.tensor(cad_vec[:, 1:], dtype=torch.long)
        cond = self.get_condition_vector(data_id)

        return {"command": command, "args": args, "id": data_id, "cond": cond}

    def _augment_vector(self, cad_vec):
        command1 = cad_vec[:, 0]
        ext_indices1 = np.where(command1 == EXT_IDX)[0]
        if len(ext_indices1) > 1 and random.random() > 0.5:
            ext_vec1 = np.split(cad_vec, ext_indices1 + 1)[:-1]

            # Sample a second vector for replacement
            second_id = random.choice(self.all_data)
            with h5py.File(os.path.join(self.raw_data, second_id + ".h5"), "r") as f:
                cad_vec2 = f["vec"][:]
            ext_indices2 = np.where(cad_vec2[:, 0] == EXT_IDX)[0]
            ext_vec2 = np.split(cad_vec2, ext_indices2 + 1)[:-1]

            n_replace = random.randint(1, min(len(ext_vec1) - 1, len(ext_vec2)))
            old_idx = sorted(random.sample(range(len(ext_vec1)), n_replace))
            new_idx = sorted(random.sample(range(len(ext_vec2)), n_replace))

            for i in range(n_replace):
                ext_vec1[old_idx[i]] = ext_vec2[new_idx[i]]

            new_vec = []
            total_len = 0
            for block in ext_vec1:
                total_len += len(block)
                if total_len > self.max_total_len:
                    break
                new_vec.append(block)
            return np.concatenate(new_vec, axis=0)

        return cad_vec

    def __len__(self):
        return len(self.all_data)


def get_dataloader_with_cond(phase, config, shuffle=None):
    is_shuffle = phase == 'train' if shuffle is None else shuffle

    dataset = CADDatasetWithCondition(phase, config, os.path.join(config.data_root, "limit_labels.json"))
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=is_shuffle, num_workers=config.num_workers,
                            worker_init_fn=np.random.seed())
    return dataloader