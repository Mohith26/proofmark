"""Proofmark: adaptive gold-question routing for crowd annotation quality control."""

from .irt import ItemBank, estimate_ability, ability_posterior, marginal_p_correct
from .detect import SPRT, kl_bernoulli, detection_information, llr_increment
from .personfit import lz, lz_from_posterior, difficulty_spread
from .router import ROUTERS, RandomRouter, FisherRouter, DetectionRouter, HybridRouter
from .simulate import make_population
from .evaluate import run_trial, run_policy, summarise, LZ_THRESHOLD

__all__ = [
    "ItemBank",
    "estimate_ability",
    "ability_posterior",
    "marginal_p_correct",
    "SPRT",
    "kl_bernoulli",
    "detection_information",
    "llr_increment",
    "lz",
    "lz_from_posterior",
    "difficulty_spread",
    "LZ_THRESHOLD",
    "ROUTERS",
    "RandomRouter",
    "FisherRouter",
    "DetectionRouter",
    "HybridRouter",
    "make_population",
    "run_trial",
    "run_policy",
    "summarise",
]
