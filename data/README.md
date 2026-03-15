# Data Directory

This directory contains datasets used in the FFSI experiments.
Files with redistribution restrictions are not included in the repository — see download instructions below.

---

## Datasets

### 1. Gene_expression_FFSI.csv
Binary gene expression data for ALL/AML leukemia classification.

- **Format**: CSV, binary features (0/1), binary label column
- **Size**: 72 samples × 3,959 features
- **Source**: Golub et al. (1999). Molecular classification of cancer: class discovery and class prediction by gene expression monitoring. *Science*, 286(5439):531–537.
- **Status**: Included in repository

---

### 2. CSF_uncorrelated_50_features.csv
50 mutually uncorrelated features (|r| < 0.3) selected from the CSF scRNA-seq dataset.

- **Format**: CSV, continuous expression values, binary label column (MS vs. Control)
- **Size**: 9,893 cells × 50 features
- **Source**: Herbst et al. (2025). Inferring single-cell and spatial microRNA activity from transcriptomics data. *Communications Biology*, 8:87.
- **Note**: Raw data was provided by the authors of Herbst et al. The preprocessed 50-feature file is included in this repository.
- **Status**: Included in repository

---

### 3. methylation_data.csv
DNA methylation data for TCGA-KIRC (kidney renal clear cell carcinoma), binarized at β = 0.5.

- **Format**: CSV, binary features (0/1), label column (binary: Stage I+II vs. III+IV; or 4-class: Stages I–IV)
- **Size**: 233 samples × 5,000 CpG sites
- **Source**: The Cancer Genome Atlas Research Network (2013). Comprehensive molecular characterization of clear cell renal cell carcinoma. *Nature*, 499(7456):43–49.
- **Status**: Not included in repository. This file was obtained from a collaborator. For access, contact the corresponding author at caplanomer@gmail.com.

---

### 4. tcga_brca_mirna_binary.csv
TCGA-BRCA miRNA-Seq expression profiles, binarized by per-feature median.

- **Format**: CSV, binary features (0/1), binary label column (1 = Primary Tumor, 0 = Solid Tissue Normal)
- **Size**: 1,200 samples × 1,881 miRNA features
- **Source**: The Cancer Genome Atlas Network (2012). Comprehensive molecular portraits of human breast tumours. *Nature*, 490(7418):61–70.
- **Portal**: https://portal.gdc.cancer.gov/projects/TCGA-BRCA
- **Status**: Not included — download via GDC API (see `download_tcga.py`)

---

### 5. tcga_brca_mrna_binary.csv
TCGA-BRCA RNA-Seq expression profiles (protein-coding genes), binarized by per-gene median.

- **Format**: CSV, binary features (0/1), binary label column (1 = Primary Tumor, 0 = Solid Tissue Normal)
- **Size**: 1,200 samples × 19,962 features
- **Source**: The Cancer Genome Atlas Network (2012). Comprehensive molecular portraits of human breast tumours. *Nature*, 490(7418):61–70.
- **Portal**: https://portal.gdc.cancer.gov/projects/TCGA-BRCA
- **Status**: Not included — download via GDC API (see `download_tcga.py`)

---

## Downloading TCGA Data

Run the provided script to download and preprocess datasets 3, 4, and 5:

```bash
python data/download_tcga.py
```

This script uses the GDC API to download the raw files and saves the processed binary CSVs to the `data/` directory. No authentication is required — all files are open-access.

Expected output:
- `data/tcga_brca_mirna_binary.csv`
- `data/tcga_brca_mrna_binary.csv`

---

## Notes

- `CSF_final_gene_matrix_for_FFSI.pkl` is an intermediate preprocessing file and is excluded from the repository. It is regenerated automatically when running the CSF experiment scripts.
- For any issues obtaining the data, contact the corresponding author at caplanomer@gmail.com.
