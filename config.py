"""Frozen configuration for the ClinVar-MVE AIGQFusion study."""
from __future__ import annotations

DATASET_NAME = "ClinVar-MVE"
REPEAT_SEEDS = (42, 2026, 3407, 7319, 9871)
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
DECISION_THRESHOLD = 0.50
CALIBRATION_TOLERANCE = 1e-4

FEATURE_VIEWS = {
    "population_frequency": ["population_AF_log10", "population_AF_popmax_log10"],
    "nucleotide_context": ["is_transition"],
    "amino_acid_impact": [
        "aa_abs_delta_hydropathy", "aa_abs_delta_volume", "aa_abs_delta_mass",
        "aa_abs_delta_charge", "aa_delta_aromatic", "aa_same_charge_class",
        "aa_same_aromatic_class", "grantham_score", "blosum62_score",
    ],
    "protein_locus_context": [
        "pos", "selected_aa_position", "protein_length", "coding_nt_length",
        "protein_relative_position",
    ],
    "gene_constraint": ["loeuf", "mis_z"],
}
FEATURES = [f for values in FEATURE_VIEWS.values() for f in values]
VIEW_INDEX = [q for q, values in enumerate(FEATURE_VIEWS.values()) for _ in values]
BIO_EDGES = ((0, 2), (1, 2), (2, 3), (3, 4), (0, 4))

# Public dataset schema. Change only these two names if a released CSV uses aliases.
LABEL_COLUMN = "label"
GROUP_COLUMN = "gene_component_id"
