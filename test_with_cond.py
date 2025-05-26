# test_with_cond.py

import argparse
from config import ConfigAE
import os
import torch
import json
import h5py
import numpy as np
from tqdm import tqdm
from cadlib.macro import *
from model import CADTransformerWithCond
from cadlib.visualize import vec2CADsolid
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Extend.TopologyUtils import TopologyExplorer

# Define material mapping and densities
MATERIAL_TABLE = {"Al": 0, "Fe": 1, "Ni": 2, "Cr": 3, "Cu": 4, "Au": 5}
INV_MATERIAL_TABLE = {v: k for k, v in MATERIAL_TABLE.items()}
DENSITY_TABLE = {
    "Al": 2700,
    "Fe": 7850,
    "Ni": 8900,
    "Cr": 7190,
    "Cu": 8960,
    "Au": 19300,
}

def generate_sample_conditions(num_samples=10):
    conds = []
    for _ in range(num_samples):
        # Select a material randomly
        mat_id = np.random.randint(0, 6)
        material_name = list(MATERIAL_TABLE.keys())[mat_id]
        density = DENSITY_TABLE[material_name]  # e.g., 2700, 7850, etc.

        # Sample volume and compute mass
        max_volume = round(np.random.uniform(0.1, 1), 4)
        max_mass = round(density * max_volume, 4)  # kg

        # Sample minimum wall thickness
        min_thick = round(np.random.uniform(0.001, 0.005), 4)  # in m

        mat_onehot = np.eye(6)[mat_id]
        cond_vec = np.concatenate([[max_volume, max_mass, min_thick], mat_onehot])
        conds.append(torch.tensor(cond_vec, dtype=torch.float32).cuda())
    return conds

def logits2vec( outputs, refill_pad=True, to_numpy=True):
    out_command = torch.argmax(torch.softmax(outputs['command_logits'], dim=-1), dim=-1)
    out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1
    if refill_pad:
        mask = ~torch.tensor(CMD_ARGS_MASK).bool().cuda()[out_command.long()]
        out_args[mask] = -1
    out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
    if to_numpy:
        out_cad_vec = out_cad_vec.detach().cpu().numpy()
    return out_cad_vec

def run_test(model_path, cfg, save_dir="generated_results", num_valid=10):
    os.makedirs(save_dir, exist_ok=True)
    model = CADTransformerWithCond(cfg).cuda()
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    model.eval()

    results = []
    valid_count = 0
    sample_attempt = 0

    pbar = tqdm(total=num_valid, desc="Generating Valid Samples")
    true_count = 0
    while valid_count < num_valid:
        sample_attempt += 1

        cond = generate_sample_conditions(1)[0]
        z = torch.randn(1, 1, cfg.dim_z).cuda()
        cond_input = cond.unsqueeze(0) if cond.dim() == 1 else cond

        try:
            with torch.no_grad():
                outputs = model(None, None, z=z, cond=cond_input, return_tgt=False)
                cad_vec = logits2vec(outputs, to_numpy=True)[0]

            # Basic validity check
            if cad_vec.shape[0] < 2 or (cad_vec[:, 0] == EXT_IDX).sum() < 1:
                raise ValueError("Missing EXT blocks")

            h5_path = os.path.join(save_dir, f"sample_{valid_count}.h5")
            with h5py.File(h5_path, 'w') as f:
                f.create_dataset("vec", data=cad_vec.astype(np.int64))

            shape = vec2CADsolid(cad_vec)
            if shape.IsNull():
                raise ValueError("Generated shape is null or invalid")

            # Volume
            props = GProp_GProps()
            brepgprop.VolumeProperties(shape, props)
            volume = props.Mass()
            if volume < 1e-3 or volume>1:
                raise ValueError(f"Invalid volume: {volume}")


            # Wall thickness
            BRepMesh_IncrementalMesh(shape, 0.001)
            faces = list(TopologyExplorer(shape).faces())
            if len(faces) < 2:
                raise ValueError("Too few faces for wall thickness analysis")

            min_dist = float("inf")
            for j in range(min(len(faces), 30)):
                for k in range(j + 1, min(len(faces), 30)):
                    d = BRepExtrema_DistShapeShape(faces[j], faces[k])
                    if d.IsDone():
                        dist = d.Value()
                        if 0 < dist < min_dist:
                            min_dist = dist
            if min_dist == float("inf"):
                raise ValueError("Wall thickness could not be computed")
            thickness = min_dist

            # Material and physical constraints
            material_logits = cond[3:]
            material_id = torch.argmax(material_logits).item()
            material_name = INV_MATERIAL_TABLE[material_id]
            density = DENSITY_TABLE[material_name]
            mass = volume * density

            max_volume = cond[0].item()
            max_mass = cond[1].item()
            min_thickness = cond[2].item()

            valid = volume < max_volume and mass < max_mass and thickness > min_thickness
            if valid:
                true_count += 1
            results.append({
                "id": f"sample_{valid_count}",
                "valid": valid,
                "volume": volume,
                "mass": mass,
                "wall_thickness": thickness,
                "material": material_name,
                "cond": {
                    "max_volume": max_volume,
                    "max_mass": max_mass,
                    "min_thickness": min_thickness,
                    "density": density
                }
            })
            valid_count += 1
            pbar.update(1)

        except Exception as e:
            continue

    pbar.close()

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Generated {valid_count} samples and {true_count} are valid.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--save_dir', type=str, default='generated_results', help='Directory to save outputs')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to generate')
    args = parser.parse_args()

    cfg = ConfigAE('test')
    run_test(model_path=args.ckpt, cfg=cfg, save_dir=args.save_dir, num_valid=args.num_samples)

if __name__ == '__main__':
    main()
