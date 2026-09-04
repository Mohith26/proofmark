"""Item response theory for gold-question calibration.

Gold questions are annotation tasks whose correct answer is already known. We
model the probability that an annotator gets one right with a 3PL IRT model,
which is the standard psychometric workhorse and, unlike plain accuracy, keeps
item difficulty and annotator skill on the same latent scale.

    P(correct | theta) = c + (1 - c) * sigmoid(a * (theta - b))

theta is annotator ability, b is item difficulty, a is discrimination, and c is
the guessing floor. For a K-way annotation task c = 1/K, because an annotator
who knows nothing still lands on the right label 1/K of the time. That floor
matters a lot here: it is the reason a sufficiently hard gold question tells you
nothing about whether someone is cheating, since honest and dishonest
annotators both bottom out at chance.
"""

from dataclasses import dataclass, field

import numpy as np

# Ability grid used for all posterior work. +/- 4 logits covers the population
# prior N(0, 1) out to four standard deviations; 161 points keeps the EAP
# quadrature error well below the sampling noise we actually care about.
THETA_GRID = np.linspace(-4.0, 4.0, 161)


def sigmoid(z):
    # Branch on sign so we never exponentiate a large positive number.
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


@dataclass
class ItemBank:
    """A pool of calibrated gold questions.

    a, b, c are parallel arrays. c is the per-item guessing floor, which is
    1/K for a K-way item.

    Everything downstream needs P(correct) for every item at every point of the
    ability grid, over and over, because the selection policies score the whole
    remaining pool on every question. That is a fixed matrix for a given bank,
    so grid_p builds it once and caches it.
    """

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    _grid_p: np.ndarray | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if not (len(self.a) == len(self.b) == len(self.c)):
            raise ValueError("a, b and c must be the same length")
        if np.any(self.a <= 0):
            raise ValueError("discrimination must be positive")
        if np.any((self.c < 0) | (self.c >= 1)):
            raise ValueError("guessing floor must be in [0, 1)")

    def __len__(self):
        return len(self.a)

    @property
    def grid_p(self):
        """P(correct) as an (n_items, n_grid) matrix, built once and cached."""
        if self._grid_p is None:
            z = self.a[:, None] * (THETA_GRID[None, :] - self.b[:, None])
            p = self.c[:, None] + (1.0 - self.c[:, None]) * sigmoid(z)
            self._grid_p = np.clip(p, 1e-9, 1.0 - 1e-9)
        return self._grid_p

    def p_correct(self, theta):
        """P(correct) for every item at a scalar ability."""
        return self.c + (1.0 - self.c) * sigmoid(self.a * (theta - self.b))

    def p_correct_item(self, idx, theta):
        z = np.array([self.a[idx] * (theta - self.b[idx])])
        return float(self.c[idx] + (1.0 - self.c[idx]) * sigmoid(z)[0])

    def fisher_information(self, theta):
        """Fisher information about theta carried by each item.

        For 3PL this is

            I(theta) = a^2 * (1 - P)/P * ((P - c)/(1 - c))^2

        which collapses to the familiar a^2 * P * (1 - P) when c = 0. It peaks
        near P = 0.5, i.e. on items matched to the annotator's own ability.
        That is exactly the property adaptive testing exploits, and exactly the
        property that makes adaptive testing bad at catching cheaters.
        """
        p = np.clip(self.p_correct(theta), 1e-9, 1 - 1e-9)
        return self.a**2 * ((1.0 - p) / p) * ((p - self.c) / (1.0 - self.c)) ** 2

    @staticmethod
    def synthetic(n_items, n_choices=4, seed=0):
        """Build a plausible bank: difficulty spread wide, discrimination modest."""
        rng = np.random.default_rng(seed)
        b = rng.uniform(-3.0, 3.0, n_items)
        a = rng.uniform(0.8, 2.0, n_items)
        c = np.full(n_items, 1.0 / n_choices)
        return ItemBank(a=a, b=b, c=c)


def log_likelihood_grid(bank, asked, correct):
    """Log P(responses | theta) evaluated on THETA_GRID.

    asked is a list of item indices, correct a list of booleans.
    """
    if not asked:
        return np.zeros_like(THETA_GRID)
    idx = np.asarray(asked, dtype=int)
    ok = np.asarray(correct, dtype=bool)
    p = bank.grid_p[idx]
    return np.where(ok[:, None], np.log(p), np.log1p(-p)).sum(axis=0)


def ability_posterior(bank, asked, correct, prior_mu=0.0, prior_sd=1.0):
    """Normalised posterior over THETA_GRID under a Gaussian population prior."""
    log_prior = -0.5 * ((THETA_GRID - prior_mu) / prior_sd) ** 2
    log_post = log_prior + log_likelihood_grid(bank, asked, correct)
    log_post -= log_post.max()
    post = np.exp(log_post)
    return post / post.sum()


def estimate_ability(bank, asked, correct, prior_mu=0.0, prior_sd=1.0):
    """Expected a-posteriori ability estimate and its posterior SD."""
    post = ability_posterior(bank, asked, correct, prior_mu, prior_sd)
    mean = float(np.sum(THETA_GRID * post))
    var = float(np.sum((THETA_GRID - mean) ** 2 * post))
    return mean, float(np.sqrt(var))


def marginal_p_correct(bank, idx, posterior):
    """P(correct on item idx) marginalised over the current ability posterior.

    We need this because the honest hypothesis is composite: we do not know the
    annotator's true ability while we are testing them. Marginalising is the
    honest way to turn it into something the sequential test can use, and the
    cost of doing so is quantified in experiments/controls.py.
    """
    return float(bank.grid_p[idx] @ posterior)


def marginal_p_correct_all(bank, posterior):
    """Vectorised marginal_p_correct over the whole bank."""
    return bank.grid_p @ posterior
