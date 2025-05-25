# Re-import necessary modules after kernel reset
import os
import json
import random
import h5py
import numpy as np
from tqdm import tqdm

from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Extend.TopologyUtils import TopologyExplorer

from cadlib.visualize import vec2CADsolid

# === Settings ===
DATA_DIR = "./cad_vec"  # update as needed
OUTPUT_JSON = "./limit_labels_new.json"
DEBUG = False  # <------ Set to False to run on full dataset

DENSITY_TABLE = {
    "Al": 2700,
    "Fe": 7850,
    "Ni": 8900,
    "Cr": 7190,
    "Cu": 8960,
    "Au": 19300,
}

def find_all_h5_files(data_dir):
    h5_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".h5"):
                h5_paths.append(os.path.join(root, file))
    return sorted(h5_paths)

def compute_volume(shape: TopoDS_Shape):
    props = GProp_GProps()
    brepgprop.VolumeProperties(shape, props)
    return props.Mass()

def compute_wall_thickness(shape: TopoDS_Shape, sample_faces=50):
    try:
        BRepMesh_IncrementalMesh(shape, 0.001)
        faces = list(TopologyExplorer(shape).faces())
        if len(faces) < 2:
            return None

        min_dist = float('inf')
        face_count = min(len(faces), sample_faces)

        for i in range(face_count):
            for j in range(i + 1, face_count):
                dist_calc = BRepExtrema_DistShapeShape(faces[i], faces[j])
                if dist_calc.IsDone():
                    dist = dist_calc.Value()
                    if 0 < dist < min_dist:
                        min_dist = dist
        return min_dist if min_dist != float('inf') else None
    except Exception as e:
        print(f"[Wall Thickness Error] {e}")
        return None

def process_h5_file(path, label_id):
    try:
        with h5py.File(path, "r") as f:
            vec = f["vec"][:]
    except Exception as e:
        return label_id, {"valid": False, "error": str(e)}

    try:
        shape = vec2CADsolid(vec)
        volume = compute_volume(shape)
        thickness = compute_wall_thickness(shape)
        material = random.choice(list(DENSITY_TABLE.keys()))
        density = DENSITY_TABLE[material]
        mass = volume * density
        return label_id, {
            "valid": True,
            "volume": round(volume, 8),
            "mass": round(mass, 3),
            "wall_thickness": round(thickness, 6) if thickness else None,
            "material": material,
            "density": density
        }
    except Exception as e:
        return label_id, {"valid": False, "error": str(e)}

# === Run ===
h5_paths = find_all_h5_files(DATA_DIR)
label_data = {}

if DEBUG:
    print("=== DEBUG MODE ===")
    path = h5_paths[0]
    label_id = os.path.splitext(os.path.basename(path))[0]
    label, data = process_h5_file(path, label_id)
    print(f"[{label_id}] → {json.dumps(data, indent=2)}")
else:
    for path in tqdm(h5_paths):
        label_id = os.path.splitext(os.path.basename(path))[0]
        label, data = process_h5_file(path, label_id)
        label_data[label] = data

    with open(OUTPUT_JSON, "w") as f:
        json.dump(label_data, f, indent=2)
