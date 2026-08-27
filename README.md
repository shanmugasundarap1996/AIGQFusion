# AIGQFusion

## Adaptive Interaction-Guided Quantum Fusion for Leakage-Controlled Missense Variant Pathogenicity Prediction

**AIGQFusion** is a hybrid classical–quantum learning framework developed for binary missense variant pathogenicity prediction.

The framework combines a strong CatBoost predictor with a biology-guided five-qubit variational quantum model and integrates their predictions through leakage-controlled cross-fitted fusion, probability calibration, and validation-gated route selection.

The accompanying **ClinVar-MVE** dataset contains 19,332 strict missense variants represented using 19 numerical predictors organized into five biological evidence views.

---

## Overview

Missense variants alter a single amino acid in a protein and may have little functional effect or may contribute to disease.

AIGQFusion evaluates whether a compact simulated quantum representation can provide complementary predictive information beyond a strong classical model while maintaining a leakage-controlled evaluation protocol.

The framework does **not** claim quantum computational superiority or hardware quantum advantage.

The main components are:

1. Multi-View Biological Evidence Mapping
2. Tuned CatBoost Classical Backbone
3. Five-Qubit Biology-Guided Variational Quantum Model
4. Cross-Fitted Classical–Quantum Evidence Integration
5. Probability Calibration
6. Validation-Gated Route Selection
7. Gene-Component-Disjoint Repeated Evaluation

---

# ClinVar-MVE Dataset

The **ClinVar Missense Variant Evidence (ClinVar-MVE)** dataset was constructed from the historical ClinVar GRCh38 VCF release dated **19 December 2020**.

### Dataset Summary

| Property | Value |
|---|---:|
| Genome assembly | GRCh38 |
| ClinVar release | 19 December 2020 |
| Total missense variants | 19,332 |
| Benign / likely benign | 12,567 |
| Pathogenic / likely pathogenic | 6,765 |
| Stored dataset fields | 51 |
| Predictive numerical features | 19 |
| Biological evidence views | 5 |
| Gene-connected components | 2,547 |
| Classification task | Binary |
| Decision threshold | 0.50 |

### Label Definition

- `0` — Benign / Likely Benign
- `1` — Pathogenic / Likely Pathogenic

Variants with uncertain, conflicting, incomplete, or other non-binary ClinVar interpretations were excluded.

---

# Data Sources

The dataset integrates information from the following public biological resources:

- **ClinVar** — variant records and clinical interpretations
- **ANNOVAR / refGene** — gene, transcript, and missense annotations
- **gnomAD v2.1.1** — population-frequency and gene-constraint evidence
- **Grantham score** — amino-acid physicochemical distance
- **BLOSUM62** — amino-acid substitution score

Only strict missense variants annotated as `nonsynonymous SNV` were retained after the frozen filtering procedure.

---

# Frozen Predictive Feature Set

The reference AIGQFusion implementation uses exactly **19 numerical predictors**.

These predictors are organized into five biological evidence views.

## View 1 — Population Frequency (`q0`)

| Feature | Description |
|---|---|
| `population_AF_log10` | Log-scale population allele-frequency evidence |
| `population_AF_popmax_log10` | Log-scale maximum population-specific allele frequency |

**Number of features:** 2

---

## View 2 — Nucleotide Context (`q1`)

| Feature | Description |
|---|---|
| `is_transition` | Indicates whether the nucleotide substitution is a transition |

**Number of features:** 1

---

## View 3 — Amino-Acid Impact (`q2`)

| Feature | Description |
|---|---|
| `aa_abs_delta_hydropathy` | Absolute hydropathy difference |
| `aa_abs_delta_volume` | Absolute amino-acid volume difference |
| `aa_abs_delta_mass` | Absolute molecular-mass difference |
| `aa_abs_delta_charge` | Absolute charge difference |
| `aa_delta_aromatic` | Change in aromatic-residue status |
| `aa_same_charge_class` | Whether reference and alternate amino acids belong to the same charge class |
| `aa_same_aromatic_class` | Whether reference and alternate residues share aromatic class |
| `grantham_score` | Grantham physicochemical substitution distance |
| `blosum62_score` | BLOSUM62 amino-acid substitution score |

**Number of features:** 9

---

## View 4 — Protein / Locus Context (`q3`)

| Feature | Description |
|---|---|
| `pos` | Genomic coordinate on GRCh38 |
| `selected_aa_position` | Amino-acid position in the selected protein |
| `protein_length` | Selected protein length |
| `coding_nt_length` | Coding-sequence length |
| `protein_relative_position` | Relative location of the substituted residue within the protein |

**Number of features:** 5

---

## View 5 — Gene Constraint (`q4`)

| Feature | Description |
|---|---|
| `loeuf` | gnomAD loss-of-function observed/expected upper-bound fraction |
| `mis_z` | gnomAD missense constraint Z-score |

**Number of features:** 2

---

## Evidence-View Summary

| View | Qubit | Features |
|---|---|---:|
| Population frequency | `q0` | 2 |
| Nucleotide context | `q1` | 1 |
| Amino-acid impact | `q2` | 9 |
| Protein/locus context | `q3` | 5 |
| Gene constraint | `q4` | 2 |
| **Total** | — | **19** |

---

# Missing Values

Missing predictor values occur only in:

- `loeuf`
- `mis_z`

Each contains **532 missing observations**.

All preprocessing in the reference evaluation is performed within the corresponding training partition.

Missing values are replaced using **training-partition medians**.

Scale-sensitive models use means and standard deviations estimated only from the corresponding training partition.

---

# Important Note on the 51 Stored Columns

The distributed ClinVar-MVE CSV contains **51 stored fields**, but only the documented **19 numerical predictors** are used for model training.

Additional columns are retained for:

- genomic identification
- annotation traceability
- gene/transcript information
- duplicate detection
- protein mapping
- group-disjoint partitioning
- dataset auditing
- reproducibility

Fields such as variant identifiers, gene names, transcript identifiers, HGVS annotations, clinical annotations, label-derived information, and audit identifiers must **not** automatically be treated as predictive features.

Researchers seeking to reproduce the reference AIGQFusion experiments should use only the documented 19-feature frozen predictor set.

---

# Leakage-Controlled Evaluation

A central design objective of AIGQFusion is to reduce information leakage between training and evaluation partitions.

The 19,332 variants were organized into:

- **2,547 gene-connected components**
- transitive multi-gene components
- identical-feature atomic groups

Twenty-seven identical complete-feature groups containing 54 variants were identified and incorporated into the grouping procedure.

The frozen data split contains:

| Partition | Variants |
|---|---:|
| Development cohort | 15,836 |
| Protected holdout | 3,496 |

The protected holdout was excluded from model-development decisions in the reported development study.

The repeated development evaluation uses:

- 5 outer repeat seeds
- 5 outer folds per repeat
- 25 outer evaluations
- 3 group-disjoint inner folds

Outer repeat seeds:

```text
42
2026
3407
7319
9871
