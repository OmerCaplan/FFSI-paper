# Forward Feature Selection Incompatibility (FFSI) in Molecular Biology Data

This repository contains the code and data for the paper:

**"How successful do we expect greedy feature selection to be in molecular biology data?"**
Omer Ella Caplan, Guy Assa, and Zohar Yakhini
*Bioinformatics*, ECCB 2026.

---

## Abstract

Forward Feature Selection (FFS) is a greedy algorithm widely used in computational biology to identify predictive feature subsets from high-dimensional molecular data. In a seminal 1977 paper Cover and Van Campenhout proved that greedy selection can fail to find the optimal subset for continuous Gaussian data. The occurrence of such failure in binary data, common in biological applications, has not been systematically studied. We define a dataset as Forward Feature Selection Incompatible (FFSI) when FFS does not recover the globally optimal feature subset, and provide an explicit construction of a dataset on which the greedy algorithm is guaranteed to fail. We then quantify FFSI prevalence empirically across five molecular biology datasets spanning microarray transcriptomics, DNA methylation, single-cell RNA sequencing, miRNA sequencing, and bulk RNA sequencing. FFSI rates vary by more than three orders of magnitude across datasets, from below 0.12% in microarray gene expression (72 instances, 3,959 features) to 33–48% in DNA methylation (233 instances, 5,000 features). Further investigating feature selection aspects, we introduce pair-greedy selection, which evaluates feature pairs rather than individual features. On the methylation data, pair-greedy selection of 6 features from a random sample of 1,000 features outperforms standard FFS on all 5,000 features in 78% of trials, yet provides no advantage under multiclass labels on the same data (2%). Our results provide a preliminary indication for what we can expect from greedy feature selection in molecular biology data.

---

## Repository Structure

```
FFSI-paper/
├── README.md                            # This file
├── requirements.txt                     # Python dependencies
├── .gitignore
│
├── src/                                 # Core library
│   ├── __init__.py
│   ├── entropy.py                       # Entropy and information gain calculations
│   └── ffsi_detection.py               # FFSI detection algorithms
│
├── data/                                # Datasets (see data/README.md)
│   ├── README.md                        # Data sources and download instructions
│   ├── download_tcga.py                 # Script to download TCGA-BRCA datasets
│   ├── Gene_expression_FFSI.csv         # ALL/AML microarray data (included)
│   └── CSF_uncorrelated_50_features.csv # CSF scRNA-seq features (included)
│
├── experiments/                         # Experiment scripts
│   ├── run_ffsi_experiment_ALL_ALM.py   # FFSI rates — ALL/AML gene expression
│   ├── run_ffsi_methylation.py          # FFSI rates — methylation (binary)
│   ├── run_ffsi_methylation_multiclass.py # FFSI rates — methylation (4-class)
│   ├── run_ffsi_csf_experiment.py       # FFSI rates — CSF scRNA-seq
│   ├── greedy_pairs_experiment_methylation.py  # Pair-greedy — methylation binary
│   ├── greedy_pairs_experiment_multiclass.py   # Pair-greedy — methylation multiclass
│   ├── greedy_pairs_vs_singles_mirna.py # Pair-greedy — BRCA miRNA-Seq
│   ├── greedy_pairs_vs_singles_mrna.py  # Pair-greedy — BRCA RNA-Seq
│   ├── greedy_ffs_full_data.py          # Standard FFS on full feature sets (g-full baseline)
│   └── ffsi_plots_publication.py        # Generate all publication figures
│
├── results/                             # Experiment outputs
│   ├── gene_expression/
│   ├── methylation/
│   ├── methylation_multiclass/
│   ├── csf/
│   └── greedy_pairs/
│
├── figures/                             # Publication figures
│   ├── fig_ffsi_rates_combined.pdf      # Figure 1 — FFSI rates (4 panels)
│   └── fig_histogram_pair_vs_full_k6.pdf # Figure 2 — pair-greedy histogram
│
└── notebooks/
    └── csf_data_exploration.ipynb       # CSF dataset exploration
```

---

## Installation

```bash
git clone https://github.com/OmerCaplan/FFSI-paper.git
cd FFSI-paper
pip install -r requirements.txt
```

---

## Reproducing Results

### 1. Obtain the data

Follow the instructions in `data/README.md`. Two datasets are included in the repository. The TCGA datasets can be downloaded by running:

```bash
python data/download_tcga.py
```

### 2. Run FFSI rate experiments

```bash
python experiments/run_ffsi_experiment_ALL_ALM.py      # ALL/AML gene expression
python experiments/run_ffsi_methylation.py             # DNA methylation (binary)
python experiments/run_ffsi_methylation_multiclass.py  # DNA methylation (4-class)
python experiments/run_ffsi_csf_experiment.py          # CSF scRNA-seq
```

Results are saved to the corresponding subfolders in `results/`.

### 3. Run pair-greedy experiments

```bash
python experiments/greedy_pairs_experiment_methylation.py
python experiments/greedy_pairs_experiment_multiclass.py
python experiments/greedy_pairs_vs_singles_mirna.py
python experiments/greedy_pairs_vs_singles_mrna.py
```

Results are saved to `results/greedy_pairs/`.

### 4. Generate figures

```bash
python experiments/ffsi_plots_publication.py
```

Figures are saved to `figures/`.

---

## Results Summary

FFSI rates across four classification tasks (R = random samples, k = selected features):

| Dataset | k=6 | k=7 | k=8 | k=9 | R |
|---|---|---|---|---|---|
| ALL/AML Microarray | 0.12% | 0.06% | 0.03% | 0.01% | 500,000 |
| Methylation (Binary) | 33% | 41% | 46% | 46% | 100,000 |
| Methylation (4-class) | 39% | 45% | 48% | 48% | 100,000 |
| CSF scRNA-seq | 4.7% | 7.5% | 12% | 18% | 100,000 |

---

## Citation

```bibtex
@article{ellacaplan2026ffsi,
  title={How successful do we expect greedy feature selection to be in molecular biology data?},
  author={Ella Caplan, Omer and Assa, Guy and Yakhini, Zohar},
  journal={Bioinformatics},
  year={2026}
}
```

---

## Contact

Corresponding author: Omer Ella Caplan (caplanomer@gmail.com)
