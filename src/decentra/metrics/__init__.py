from .prediction import prediction_fidelity
from .attribution import (
    topk,
    advtopk,
    advfull,
    attribution_fidelity,
    random_baseline_advtopk,
)
from .interventional import (
    interventional_fidelity,
    median_intervention_fidelity,
    compute_sic_sc,          # legacy alias
    extract_bin_structure,
)
from .named import (
    align_attributions,
    AlignmentInfo,
    topk_named,
    advtopk_named,
    advfull_named,
    random_baseline_advtopk_named,
    attribution_fidelity_named,
)

__all__ = [
    "prediction_fidelity",
    "topk",
    "advtopk",
    "advfull",
    "attribution_fidelity",
    "random_baseline_advtopk",
    "interventional_fidelity",
    "median_intervention_fidelity",
    "compute_sic_sc",
    "extract_bin_structure",
    "align_attributions",
    "AlignmentInfo",
    "topk_named",
    "advtopk_named",
    "advfull_named",
    "random_baseline_advtopk_named",
    "attribution_fidelity_named",
]
