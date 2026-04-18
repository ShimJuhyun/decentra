"""decentra — Interpretable surrogate modeling for black-box credit scoring."""

__version__ = "0.1.0"

from .scorecard import Scorecard
from .scorecard_model import ScorecardModel
from .stats import TrainingStats, FeatureStats

__all__ = ["Scorecard", "ScorecardModel", "TrainingStats", "FeatureStats"]
