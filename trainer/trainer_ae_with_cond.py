import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import os
import h5py
from model import CADTransformerWithCond
from .base import BaseTrainer
from .loss import CADLoss
from .scheduler import GradualWarmupScheduler
from cadlib.macro import *
from cadlib.visualize import vec2CADsolid
from OCC.Core.BRepGProp import brepgprop
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRepExtrema import BRepExtrema_DistShapeShape
from OCC.Extend.TopologyUtils import TopologyExplorer
import json
from model.regressor import PhysicalRegressorFromLogits


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

class TrainerAEWithCond(BaseTrainer):
    def build_net(self, cfg):
        self.net = CADTransformerWithCond(cfg).cuda()

    def set_optimizer(self, cfg):
        self.optimizer = optim.Adam(self.net.parameters(), cfg.lr)
        self.scheduler = GradualWarmupScheduler(self.optimizer, 1.0, cfg.warmup_step)

    def set_loss_function(self):
        regressor = PhysicalRegressorFromLogits(self.cfg).cuda()
        ckpt_path = "proj_log/regressor_logits/model/latest.pth"
        assert os.path.exists(ckpt_path), f"Regressor checkpoint not found at {ckpt_path}"

        regressor.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
        regressor.eval()
        self.loss_func = CADLoss(self.cfg,physical_regressor=regressor).cuda()

    def forward(self, data):
        commands = data['command'].cuda()
        args = data['args'].cuda()
        cond = data['cond'].cuda()
        outputs = self.net(commands, args, cond=cond)
        loss_dict = self.loss_func(outputs)
        if loss_dict is None:
            return None, None
        return outputs, loss_dict

    def encode(self, data, is_batch=False):
        commands = data['command'].cuda()
        args = data['args'].cuda()
        cond = data['cond'].cuda()
        if not is_batch:
            commands = commands.unsqueeze(0)
            args = args.unsqueeze(0)
            cond = cond.unsqueeze(0)
        z = self.net(commands, args, cond=cond, encode_mode=True)
        return z

    def decode(self, z, cond):
        outputs = self.net(None, None, z=z, cond=cond, return_tgt=False)
        return outputs

    def logits2vec(self, outputs, refill_pad=True, to_numpy=True):
        out_command = torch.argmax(torch.softmax(outputs['command_logits'], dim=-1), dim=-1)
        out_args = torch.argmax(torch.softmax(outputs['args_logits'], dim=-1), dim=-1) - 1
        if refill_pad:
            mask = ~torch.tensor(CMD_ARGS_MASK).bool().cuda()[out_command.long()]
            out_args[mask] = -1
        out_cad_vec = torch.cat([out_command.unsqueeze(-1), out_args], dim=-1)
        if to_numpy:
            out_cad_vec = out_cad_vec.detach().cpu().numpy()
        return out_cad_vec

    def evaluate(self, test_loader):
        self.net.eval()
        pbar = tqdm(test_loader)
        pbar.set_description("EVALUATE[{}]".format(self.clock.epoch))
        eval_dir = os.path.join(self.cfg.proj_dir, "eval_gen_h5")
        os.makedirs(eval_dir, exist_ok=True)

        results = []

        for i, data in enumerate(pbar):
            with torch.no_grad():
                cond = data['cond'].cuda()
                z = self.encode(data, is_batch=True)
                outputs = self.decode(z, cond)
                cad_vec = self.logits2vec(outputs, to_numpy=True)[0]  # Assume batch size = 1

            # Save to h5
            data_id = data['id'][0]
            h5_path = f"tmp/{data_id}_gen.h5"
            os.makedirs(os.path.dirname(h5_path), exist_ok=True)

            with h5py.File(h5_path, 'w') as f:
                f.create_dataset("vec", data=cad_vec.astype(np.int64))

            try:
                shape = vec2CADsolid(cad_vec)
                # Compute volume
                props = GProp_GProps()
                brepgprop.VolumeProperties(shape, props)
                volume = props.Mass()
                # Compute wall thickness
                BRepMesh_IncrementalMesh(shape, 0.001)
                faces = list(TopologyExplorer(shape).faces())
                min_dist = float('inf')
                for j in range(min(len(faces), 30)):
                    for k in range(j + 1, min(len(faces), 30)):
                        d = BRepExtrema_DistShapeShape(faces[j], faces[k])
                        if d.IsDone() and d.Value() < min_dist and d.Value() > 0:
                            min_dist = d.Value()
                thickness = min_dist if min_dist != float('inf') else None
                # Compute mass
                material_logits = cond[0, 3:]  # shape: (6,)
                material_id = torch.argmax(material_logits).item()
                material_name = INV_MATERIAL_TABLE[material_id]
                density = DENSITY_TABLE[material_name]

                # Recompute mass
                mass = volume * density
                # Check constraints
                max_volume = cond[0, 0].item()
                max_mass = cond[0, 1].item()
                min_thickness = cond[0, 2].item()
                valid = volume < max_volume and mass < max_mass and thickness is not None and thickness > min_thickness
                results.append({
                    "id": data_id,
                    "valid": valid,
                    "volume": volume,
                    "mass": mass,
                    "wall_thickness": thickness,
                    "cond": {
                        "max_volume": max_volume,
                        "max_mass": max_mass,
                        "min_thickness": min_thickness,
                        "density": density
                    }
                })
            except Exception as e:
                results.append({"id": data_id, "valid": False, "error": str(e)})

        # Print summary
        valid_count = sum(r["valid"] for r in results if r.get("valid") is not None)
        print(f"[VALID] {valid_count}/{len(results)}")
        results_path = os.path.join(eval_dir, f"results_epoch_{self.clock.epoch}.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        if os.path.exists(h5_path):
            os.remove(h5_path)

