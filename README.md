# DAE-FIMMCA

Reference implementation of the paper:

> **Deep AutoEncoder Fuzzy Inclusive Monotone Multi-View Correlation Analysis for Image Recognition**

This repository is built on top of the original DCCAE implementation. DAE-FIMMCA uses the
**Fuzzy Inclusive Monotone (FIMMCA)** relationship as a similarity measure, trains two deep
autoencoders with a contrastive objective, and performs classification directly in the latent
space **without any additional classifier**.

---

## 1. Requirements

| Item | Value |
|---|---|
| Python | 3.8.20 |
| Main dependencies | `torch` 2.4.1, `numpy` 1.24.3, `scikit-learn` 1.3.0, `opencv-python` 4.10.0, `cca_zoo` 2.6.0 |
| Hardware used in the paper | Intel Core i5-12400F CPU + NVIDIA GeForce RTX 4060 GPU |

**The repository root must be on `PYTHONPATH`**, because the scripts import each other:

```bash
# Linux / macOS
export PYTHONPATH=/path/to/DAE-FIMMCA

# Windows (cmd)
set PYTHONPATH=D:\path\to\DAE-FIMMCA

# Windows (PowerShell)
$env:PYTHONPATH="D:\path\to\DAE-FIMMCA"
```

---

## 2. Repository Structure

### 2.1 Core algorithm modules (bottom-up dependency order)

```
fimdmf_ops.py      Differentiable FIMMCA operators
                   soft_rank / soft_interval / dfimdmf / hard_fimdmf
                   -- corresponds to Sec. 3.2 (soft ranking, soft interval,
                      soft min/max) and the STE loss classes
        |
        v
fimdmf_fast.py     Vectorised pairwise soft FIMMCA (fimdmf_pair_fast)
                   -- numerically equivalent to the loop version (~1e-16),
                      about 2.9x faster
        |
        v
fimdmf_core.py     Shared building blocks
                   fimdmf_pair / Encoder / rank_matrix / classify / prepare_train_blocks
                   -- classify implements hard FIMMCA-1NN (Sec. 3.3)
        |
        v
dae_fimmca.py      * Main method: multi-view self-supervised contrastive training
                     (the complete model described in the paper)
```

### 2.2 Experiment scripts (mapped to the paper)

| File | Paper item | Description |
|---|---|---|
| `exp_1_comparison.py` | **Tables 4 and 5** | Accuracy of seven methods (CCA, RCCA, GRCCA, DCCA, DCCAE, ConvCCA, DAE-FIMMCA) on five databases |
| `exp_2_noise_sweep.py` | **Fig. 4** | Accuracy versus corruption ratio on USPS under salt-and-pepper noise |
| `exp_3_ablation.py` | **Table 6, rows 2 and 3** | Row 2: MSE objective (plain autoencoder) + FIMMCA-1NN. Row 3: FIMMCA objective + SVM |
| `exp_4_time_cost.py` | **Fig. 5** | Training + encoding time of the seven methods |
| `make_datasets.py` | Data preparation | Derives the clean USPS set and the clean-and-rotated variant from `usps_view1.gz` |

### 2.3 Original files

`DeepFCAModel.py`, `objectives.py`, `linear_cca.py`, `DeepFCA.py`, `DeepAEFCA.py`,
`DeepAEFCA02.py`, `ConvCCA.py`, `ConvFCA.py`, and `utilssss.py` are inherited from the
original DCCAE code base and are kept unchanged in substance.

Relative to the original DCCAE implementation, only minor compatibility fixes were applied
(GPU-safe tensor conversion and updated module import paths).

---

## 3. Quick Start

```bash
# Main method
python dae_fimmca.py

# Tables 4 and 5: method comparison
python exp_1_comparison.py

# Fig. 4: noise sweep
python exp_2_noise_sweep.py

# Table 6: ablation
python exp_3_ablation.py

# Fig. 5: time cost
python exp_4_time_cost.py

# Data preparation (clean USPS, clean + rotated USPS)
python make_datasets.py
```

---

## 4. Datasets

The data are stored as pickled `.gz` files under the `test_with_*/` directories. Each file is a
list of three splits, `[train, val, test]`, where every split is a pair `[features, labels]`.

> **Note.** The archives are about 1.7 GB in total and are therefore **not** tracked in this
> repository. They are available on Zenodo:
>
> - Record: <https://zenodo.org/records/23468128>
> - DOI: `10.5281/zenodo.23468128`
>
> Once downloaded, extract them so that the directory layout is
> `DAE-FIMMCA/test_with_usps/`, `DAE-FIMMCA/test_with_noisymnist/`, and so on.

### 4.1 Directory-to-database mapping

| Database | Directory | File prefix | Dimension | Classes |
|---|---|---|---|---|
| MNIST | `test_with_noisymnist/` | `my_noisymnist_*` | 784 (28x28) | 10 |
| Fashion-MNIST | `test_with_fashionmnist/` | `fashionmnist_*` | 784 (28x28) | 10 |
| USPS | `test_with_usps/` | `usps_*` | 256 (16x16) | 10 |
| COIL100 | `test_with_coil100/` | `coil_100_*` | 1024 (32x32) | 100 |
| CMU PIE | `test_cmu/` | **`cmu_lu_*`** | 1024 (32x32) | **68** |
| Extended Yale B | `test_extended/` | **`pie_*_fmmca`** | 2304 (48x48) | **38** |

> **Note on file naming.** The file prefix does not always match the directory name: files
> inside `test_extended/` are prefixed with `pie_*`, while those inside `test_cmu/` are prefixed
> with `cmu_lu_*`. **The directory names are authoritative**, which can be verified by the number
> of classes (68 subjects for CMU PIE, 38 for Extended Yale B).

### 4.2 Noise variants

Each directory contains four noise variants, distinguished by the file suffix:

| Suffix | Noise type |
|---|---|
| none | salt and pepper |
| `_new` | independent random noise |
| `_speckle` | speckle |
| `_white_mask` | block occlusion (5x5) |

### 4.3 Data conventions

- `view1` is the clean image, normalised to [0, 1]; **all noise is applied to `view2` only**.
- `view1` is used for evaluation; `view2` participates in training only.
- Within one variant, `view1[i]` and `view2[i]` originate from the same source image, so the
  labels are aligned.
- In the base variant (no suffix), the salt-and-pepper noise sets 10% of the pixels to salt and
  another 10% to pepper, i.e. **20% of the pixels are corrupted**, which corresponds to a
  corruption ratio of ρ = 0.2 in the paper.

---

## 5. Implementation Notes

The following design decisions are essential for reproducing the method.

1. **The hard FIMMCA formulation is non-differentiable** because it involves sorting and set
   intersection. Training therefore uses a **differentiable surrogate** (soft ranking, soft
   interval indicator, and log-sum-exp soft min/max), whereas **evaluation still uses the hard
   FIMMCA**.

2. **Maximising the cross-view correlation alone leads to representation collapse**: the encoders
   can trivially attain the maximum by mapping every sample to the same vector. The training
   objective is therefore an **InfoNCE-style contrastive loss**, in which the two views of the
   same sample form a positive pair and different samples form negative pairs.

3. **The learning rate must be small.** A large learning rate causes the latent representation to
   collapse within a single epoch: the encoder output is destroyed and the accuracy drops sharply.
   A conservative learning rate is therefore used throughout.

4. **The latent dimension matters.** The number of partition sizes scanned by FIMMCA grows with the
   latent dimension. When the latent space is too small, only a few partition sizes are available,
   and the discriminative power of the measure degrades substantially.

5. **Encoder parameter sharing.** By default the two encoders share parameters, i.e. a single
   network is applied to both views. This setting is important for optimisation: with two
   independently initialised networks the cross-view signal at initialisation is very weak
   (the FIMMCA gap between positive and negative pairs is about 0.006, with a signal-to-noise
   ratio of roughly 0.15), and training does not start. Sharing the parameters raises the same
   gap to about 0.059 (signal-to-noise ratio about 1.5). The flag `--separate` restores the
   two-independent-encoder setting for comparison.

---

## 6. Additional Notes

- **Evaluation cost.** `classify` implements FIMMCA-1NN, whose complexity is
  O(number of test samples x number of reference samples x dimension). For large databases such
  as MNIST and Fashion-MNIST, subsampling via `--eval-train` / `--eval-test` is recommended.
- **Reproducible tie-breaking.** FIMMCA requires sorting the feature values, and the data contain
  many ties (for example, the large number of zero-valued pixels in MNIST). The implementation
  uses a stable sort, so ties are broken in ascending order of the original index.
- **Harmless console output.** Training prints a `torch.set_default_tensor_type` deprecation
  warning and a short line of debug output from `utilssss.py`. Neither affects the results.

---

## 7. Data Availability

The datasets used in this study are publicly available on Zenodo:

- Record: <https://zenodo.org/records/23468128>
- DOI: [10.5281/zenodo.23468128](https://doi.org/10.5281/zenodo.23468128)

---

## 8. Citation

If you use this code or the accompanying data, please cite:

```bibtex
@unpublished{liang2026daefimmca,
  title  = {Deep AutoEncoder Fuzzy Inclusive Monotone Multi-View Correlation Analysis
            for Image Recognition},
  author = {Liang, Jin and Li, Xin and Zhong, Haishu},
  note   = {Manuscript under review},
  year   = {2026}
}
```

> The entry above will be updated with the journal name and volume once the manuscript
> is accepted.
