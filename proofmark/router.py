"""Gold-question selection policies.

This is where the interesting part lives. Every policy answers the same
question -- which gold question do we spend next on this annotator -- but they
optimise different things, and the whole point of the project is that the two
obvious objectives pull in opposite directions.

    Fisher information (measure their skill accurately) peaks on items the
    annotator gets right about half the time.

    Detection information (work out whether they are cheating) peaks on items
    the annotator gets right almost always, because that is where an honest
    person and a chance-level guesser look most different.

So a textbook adaptive test, which maximises Fisher information, spends its
whole budget in precisely the region where cheating is hardest to see.
experiments/run_benchmark.py measures how much that costs.
"""

import numpy as np

from .irt import THETA_GRID, marginal_p_correct_all


def _kl_vec(p, q):
    """Elementwise KL(Bern(p) || Bern(q)) in nats."""
    p = np.clip(p, 1e-9, 1 - 1e-9)
    q = np.clip(q, 1e-9, 1 - 1e-9)
    return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))


def _unit(x):
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


class Router:
    """Base class. Subclasses implement score()."""

    name = "base"

    def __init__(self, bank, rng=None, top_k=1, epsilon=0.0):
        self.bank = bank
        self.rng = rng if rng is not None else np.random.default_rng(0)
        # Sampling from the top_k scoring items instead of always taking the
        # argmax. This makes the policy non-deterministic, which sounds like it
        # should defeat an annotator working from a memorised answer key. The
        # gold-leak experiment shows it does not, because the top k items are
        # all drawn from the same easy corner of the bank that the adversary
        # memorised in the first place.
        self.top_k = top_k
        # What does help: spending a fraction of the budget on a uniformly
        # random item regardless of score. That is the only way difficulty
        # spread gets into the response pattern, and without spread the
        # person-fit statistic in personfit.py has nothing to detect.
        self.epsilon = epsilon

    def score(self, available, posterior, sprt):
        raise NotImplementedError

    def select(self, available, posterior, sprt):
        if len(available) == 0:
            raise ValueError("no gold questions left")
        if self.epsilon > 0 and self.rng.random() < self.epsilon:
            return int(self.rng.choice(available))
        scores = self.score(available, posterior, sprt)
        k = min(self.top_k, len(available))
        # argpartition gives us the k best without a full sort.
        top = np.argpartition(-scores, k - 1)[:k]
        return int(available[self.rng.choice(top)])



class RandomRouter(Router):
    """Uniformly random gold question. The policy most platforms actually run."""

    name = "random"

    def score(self, available, posterior, sprt):
        return self.rng.random(len(available))


class FisherRouter(Router):
    """Classic computerised adaptive testing: maximise information about ability."""

    name = "fisher"

    def score(self, available, posterior, sprt):
        theta_hat = float(THETA_GRID @ posterior)
        return self.bank.fisher_information(theta_hat)[available]


class DetectionRouter(Router):
    """Maximise expected drift of the sequential honesty test.

    Under H_honest the log-likelihood ratio drifts up by KL(honest || spam) per
    question; under H_spam it drifts down by KL(spam || honest). Weighting the
    two by the current belief gives the expected absolute drift, which is what
    we want to maximise if the goal is to settle the question fast in whichever
    direction the truth lies.
    """

    name = "detection"

    def score(self, available, posterior, sprt):
        pi = sprt.posterior_honest
        ph = marginal_p_correct_all(self.bank, posterior)[available]
        ps = self.bank.c[available]
        return pi * _kl_vec(ph, ps) + (1.0 - pi) * _kl_vec(ps, ph)


class HybridRouter(Router):
    """Spend on detection while honesty is open, on ability once it is settled.

    The mixing weight is the SPRT's own resolution: how far the log-likelihood
    ratio has travelled toward a decision boundary, in [0, 1]. Early on that is
    near zero and we behave like DetectionRouter. Once the test is close to
    calling it, the weight goes to one and we behave like a normal adaptive
    test. Nothing has to be tuned by hand, which is the main reason to prefer
    it over a fixed blend.
    """

    name = "hybrid"

    def __init__(self, bank, rng=None, top_k=3, epsilon=0.0):
        super().__init__(bank, rng, top_k, epsilon)

    def score(self, available, posterior, sprt):
        lam = sprt.resolution
        theta_hat = float(THETA_GRID @ posterior)

        fisher = self.bank.fisher_information(theta_hat)[available]

        pi = sprt.posterior_honest
        ph = marginal_p_correct_all(self.bank, posterior)[available]
        ps = self.bank.c[available]
        detect = pi * _kl_vec(ph, ps) + (1.0 - pi) * _kl_vec(ps, ph)

        # Normalise each objective to [0, 1] before mixing. They are in
        # different units (reciprocal-variance vs nats) so a raw sum would be
        # dominated by whichever happens to be numerically larger.
        return lam * _unit(fisher) + (1.0 - lam) * _unit(detect)


ROUTERS = {
    "random": RandomRouter,
    "fisher": FisherRouter,
    "detection": DetectionRouter,
    "hybrid": HybridRouter,
}
