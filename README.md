# DeepCAD with Conditional Constraints

This repository is a **modification** of the original [DeepCAD](https://github.com/ChrisWu1997/DeepCAD.git) project, introducing **conditional training** for constraint-aware generative CAD modeling. We build on DeepCAD's vector-based CAD autoencoder and enable generation under physical and manufacturability constraints such as **volume**, **mass**, and **wall thickness**.

## 🆕 Key Modifications

* ✳️ **Conditional Variational Autoencoder (VAE)** replacing the original autoencoder.
* 📦 `train_with_cond.py`: trains the VAE under constraint supervision.
* 🧱 Physical constraints used:

  * Maximum Volume
  * Maximum Mass
  * Minimum Wall Thickness
  * Material selection (Al, Fe, Ni, Cr, Cu, Au) with known densities
* 🔍 `label_limit.py`: used to generate the conditional vectors (volume, mass, thickness, material).
* 🧠 Trained a `PhysicalPropertyRegressor` to predict physical quantities from **command and argument logits**, enabling **differentiable constraint loss**.
* ✅ Manufacturability checked via:

  * Shape validity (`vec2CADsolid`)
  * OpenCascade mesh building
  * Face distance estimation for wall thickness

## 🧪 Testing with Constraint-Aware Sampling

We provide `test_with_cond.py` to sample from the latent space under user-defined constraints:

```bash
python test_with_cond.py \
    --ckpt YOUR_PATH
```

It:

* Samples from the latent space
* Applies constraints from randomly sampled condition vectors
* Converts generated logits to CAD vectors
* Reconstructs shapes and checks physical validity (mass, volume, thickness)
* Saves `.h5` files and evaluation results in JSON

## 🚀 How to Train

Make sure you've installed dependencies as per the original repo. Then:

```bash
python train_with_cond.py \
    --exp_name YOUR_NAME \
    --data_root YOUR_ROOT
```

## 📥 Pretrained Models

We provide pretrained weights for both the **constraint-aware VAE** and the **physical regressor** for convenience. You can download them from the following Google Drive folder:

🔗 **[Download Pretrained Models](https://drive.google.com/drive/folders/1-_eXKat98WlUOyAyBk5WBxHfGZjeJUPN?usp=sharing)**

Contents include:

* `latest.pth` – the latest checkpoint of the conditional VAE (`CADTransformerWithCond`)
* `regressor.pth` – the physical property regressor used during training and evaluation

To use:

* Load the VAE model in `train_with_cond.py` or `test_with_cond.py` via `--ckpt`
* Modify the `set_loss_function()` method to load the regressor for constraint loss


The training pipeline includes:

* Encoder + Conditional VAE bottleneck
* Decoder with condition-aware transformer
* Physical regressor used during training for soft constraint supervision

### 🔧 Data Requirements

Downloaded from the original [DeepCAD](https://github.com/ChrisWu1997/DeepCAD.git).

* Vectorized CAD sequences under `data/cad_vec/`
* Corresponding physical metadata JSON generated via `label_limit.py`

## 🔍 Physical Regressor

The physical regressor maps decoded **command logits** and **argument logits** to:

* Volume
* Wall Thickness

Trained separately and loaded into the loss module during conditional VAE training.

## 📁 Project Structure (New Additions)

```
DeepCAD/
│
├── train_with_cond.py        # Train conditional VAE
├── test_with_cond.py         # Sample and test conditioned generations
├── model/
│   ├── autoencoder.py        # Conditional encoder, bottleneck, decoder
│   ├── regressor.py # Predict physical properties from logits
│
├── trainer/
│   ├── trainer_ae_with_cond.py
│   ├── trainer_regressor.py
│   ├── loss.py               # Includes differentiable constraint loss
│
└── label_limit.py        # Generates constraint labels
```

## 📦 Citation

If you use this extension of DeepCAD, please also cite the original:

```bibtex
@InProceedings{Wu_2021_ICCV,
    author    = {Wu, Rundi and Xiao, Chang and Zheng, Changxi},
    title     = {DeepCAD: A Deep Generative Network for Computer-Aided Design Models},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year      = {2021},
    pages     = {6772-6782}
}
```