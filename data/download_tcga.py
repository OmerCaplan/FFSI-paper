"""
Download and binarize TCGA-BRCA miRNA-Seq and RNA-Seq data from the GDC API.

Data source: The Cancer Genome Atlas Network. Comprehensive molecular portraits
of human breast tumours. Nature, 490(7418):61-70, 2012.
Portal: https://portal.gdc.cancer.gov/projects/TCGA-BRCA

Output files (saved to data/):
  - tcga_brca_mirna_binary.csv   (1,200 samples x 1,881 features + label)
  - tcga_brca_mrna_binary.csv    (1,200 samples x ~19,962 protein-coding features + label)

Labels: 1 = Primary Tumor, 0 = Solid Tissue Normal

Usage:
  python data/download_tcga.py

Requirements:
  pip install requests pandas numpy
"""

import requests
import json
import pandas as pd
import tarfile
import os
import time

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT  = "https://api.gdc.cancer.gov/data"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
BATCH_SIZE = 100


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def query_gdc(filters):
    params = {
        "filters": json.dumps(filters),
        "fields": "file_id,file_name,cases.samples.sample_type",
        "format": "JSON",
        "size": 2000,
    }
    r = requests.get(GDC_FILES_ENDPOINT, params=params)
    r.raise_for_status()
    return r.json()["data"]["hits"]


def build_metadata(hits):
    records = []
    for h in hits:
        fid   = h["file_id"]
        fname = h["file_name"]
        for case in h.get("cases", []):
            for samp in case.get("samples", []):
                st = samp.get("sample_type", "")
                if "Primary Tumor" in st:
                    label = 1
                elif "Solid Tissue Normal" in st:
                    label = 0
                else:
                    continue
                records.append({"file_id": fid, "file_name": fname, "label": label})
    return pd.DataFrame(records).drop_duplicates("file_id")


def download_batches(file_ids, parse_fn):
    """Download file_ids in batches and parse each batch with parse_fn."""
    all_samples = {}
    for i in range(0, len(file_ids), BATCH_SIZE):
        batch = file_ids[i:i + BATCH_SIZE]
        print(f"  Batch {i // BATCH_SIZE + 1}/{(len(file_ids) - 1) // BATCH_SIZE + 1} ({len(batch)} files)...")
        for attempt in range(3):
            try:
                response = requests.post(
                    GDC_DATA_ENDPOINT,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"ids": batch}),
                    stream=True,
                    timeout=120,
                )
                tar_path = os.path.join(OUTPUT_DIR, f"_tmp_batch_{i}.tar.gz")
                with open(tar_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                with tarfile.open(tar_path, "r:gz") as tar:
                    parse_fn(tar, all_samples)
                os.remove(tar_path)
                break
            except Exception as e:
                print(f"    Attempt {attempt + 1} failed: {e}")
                time.sleep(5)
        time.sleep(1)
    return all_samples


def attach_labels_and_save(all_samples, df_meta, out_path):
    expr = pd.DataFrame(all_samples).T
    print(f"  Shape: {expr.shape}")
    medians = expr.median(axis=0)
    binary = (expr > medians).astype(int)
    id_to_label = dict(zip(df_meta["file_id"], df_meta["label"]))
    binary["label"] = binary.index.map(id_to_label)
    binary = binary.dropna(subset=["label"])
    binary["label"] = binary["label"].astype(int)
    print(f"  Final shape: {binary.shape}")
    print(binary["label"].value_counts().to_string())
    binary.to_csv(out_path)
    print(f"  Saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# miRNA-Seq
# ─────────────────────────────────────────────────────────────────────────────

def parse_mirna_batch(tar, all_samples):
    for member in tar.getmembers():
        if member.name.endswith(".txt") and "mirnas.quantification" in member.name.lower():
            f = tar.extractfile(member)
            if f is None:
                continue
            tmp = pd.read_csv(f, sep="\t")
            tmp = tmp[["miRNA_ID", "reads_per_million_miRNA_mapped"]].set_index("miRNA_ID")
            sample_id = member.name.split("/")[0]
            all_samples[sample_id] = tmp["reads_per_million_miRNA_mapped"]


def download_mirna():
    print("=" * 60)
    print("Downloading TCGA-BRCA miRNA-Seq...")
    print("=" * 60)
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-BRCA"}},
            {"op": "=", "content": {"field": "data_type",                "value": "miRNA Expression Quantification"}},
            {"op": "=", "content": {"field": "access",                   "value": "open"}},
        ],
    }
    hits    = query_gdc(filters)
    df_meta = build_metadata(hits)
    print(f"  Files found: {len(hits)}, samples: {len(df_meta)}")
    all_samples = download_batches(df_meta["file_id"].tolist(), parse_mirna_batch)
    print(f"  Total samples parsed: {len(all_samples)}")
    out_path = os.path.join(OUTPUT_DIR, "tcga_brca_mirna_binary.csv")
    attach_labels_and_save(all_samples, df_meta, out_path)


# ─────────────────────────────────────────────────────────────────────────────
# RNA-Seq (mRNA)
# ─────────────────────────────────────────────────────────────────────────────

def parse_mrna_batch(tar, all_samples):
    for member in tar.getmembers():
        if not (member.name.endswith(".tsv") and "augmented_star_gene_counts" in member.name.lower()):
            continue
        f = tar.extractfile(member)
        if f is None:
            continue
        try:
            tmp = pd.read_csv(f, sep="\t", skiprows=1)
            tmp = tmp[tmp["gene_type"] == "protein_coding"].copy()
            tmp = tmp[["gene_name", "tpm_unstranded"]].dropna()
            tmp = tmp.set_index("gene_name")["tpm_unstranded"]
            sample_id = member.name.split("/")[0]
            all_samples[sample_id] = tmp
        except Exception:
            continue


def download_mrna():
    print("=" * 60)
    print("Downloading TCGA-BRCA RNA-Seq (mRNA)...")
    print("=" * 60)
    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id",  "value": "TCGA-BRCA"}},
            {"op": "=", "content": {"field": "data_type",                 "value": "Gene Expression Quantification"}},
            {"op": "=", "content": {"field": "experimental_strategy",     "value": "RNA-Seq"}},
            {"op": "=", "content": {"field": "data_format",               "value": "TSV"}},
            {"op": "=", "content": {"field": "access",                    "value": "open"}},
        ],
    }
    hits    = query_gdc(filters)
    df_meta = build_metadata(hits)
    print(f"  Files found: {len(hits)}, samples: {len(df_meta)}")
    all_samples = download_batches(df_meta["file_id"].tolist(), parse_mrna_batch)
    print(f"  Total samples parsed: {len(all_samples)}")
    out_path = os.path.join(OUTPUT_DIR, "tcga_brca_mrna_binary.csv")
    attach_labels_and_save(all_samples, df_meta, out_path)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    download_mirna()
    download_mrna()
    print("\nDone. Both datasets saved to data/")
