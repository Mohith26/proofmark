"""Person-fit statistics, for the cheaters the sequential test cannot see.

The SPRT in detect.py asks one question: is this person responding at chance?
That catches a spammer clicking randomly. It structurally cannot catch someone
who answers easy items correctly and guesses on the rest, because that is not
chance responding -- it is exactly what a mediocre but honest annotator looks
like. Aggregate accuracy is the same, so a likelihood ratio built on aggregate
accuracy has nothing to work with. The benchmark measures this blind spot
rather than glossing over it: recall on a gold-leak adversary is 0.00.

What does separate them is the *shape* of the response pattern. An honest
annotator's mistakes are spread around their ability level the way the IRT
model says they should be. Someone working off a memorised answer key is
perfect above a cutoff and at chance below it, with no transition in between.
That is a model-fit question, and psychometrics has a standard tool for it:
the standardised log-likelihood person-fit index lz (Drasgow, Levine and
Williams, 1985).

    l0    = sum_i [ x_i log P_i + (1 - x_i) log(1 - P_i) ]
    E[l0] = sum_i [ P_i log P_i + (1 - P_i) log(1 - P_i) ]
    V[l0] = sum_i P_i (1 - P_i) [ log(P_i / (1 - P_i)) ]^2
    lz    = (l0 - E[l0]) / sqrt(V[l0])

lz is approximately standard normal for a response pattern that fits the
model. Strongly negative values mean the pattern is less likely than the model
expects, which is the signature of aberrant responding.

One caveat worth stating plainly: lz is only informative if the items actually
span a range of difficulty. If every question asked was trivially easy, a
memorised answer key fits the model perfectly well. So this only works in
combination with a router that deliberately spends part of its budget away
from the detection optimum -- see the epsilon parameter on Router.
"""

import math

import numpy as np

from .irt import THETA_GRID, estimate_ability


def lz(bank, asked, correct, theta_hat):
    """Standardised person-fit index at a point ability estimate.

    Returns 0.0 when there is not enough variance to standardise against,
    which happens if every item asked was effectively deterministic.
    """
    if len(asked) == 0:
        return 0.0

    idx = np.asarray(asked, dtype=int)
    x = np.asarray(correct, dtype=float)
    p = np.clip(bank.p_correct(theta_hat)[idx], 1e-6, 1 - 1e-6)

    log_p, log_q = np.log(p), np.log1p(-p)
    l0 = float(np.sum(x * log_p + (1 - x) * log_q))
    expected = float(np.sum(p * log_p + (1 - p) * log_q))
    variance = float(np.sum(p * (1 - p) * (log_p - log_q) ** 2))

    if variance < 1e-9:
        return 0.0
    return (l0 - expected) / math.sqrt(variance)


def lz_from_posterior(bank, asked, correct, posterior):
    """lz evaluated at the posterior mean ability."""
    return lz(bank, asked, correct, float(THETA_GRID @ posterior))


def difficulty_spread(bank, asked):
    """Standard deviation of the difficulties actually asked.

    Reported alongside the fit statistics because both are meaningless without
    spread, and it is the number that shows why pure detection routing defeats
    its own fit check: it asks the same easy corner of the bank every time.
    """
    if len(asked) < 2:
        return 0.0
    return float(np.std(bank.b[np.asarray(asked, dtype=int)]))


def split_difficulty_z(bank, asked, correct, min_per_side=4, prior_sd=2.0):
    """Does this annotator have the same ability on easy items as on hard ones?

    lz turned out to have essentially no power against a memorised answer key
    (the benchmark measures 0.00 recall). The reason is that lz standardises
    against the model evaluated at the *fitted* ability, and the aberrant
    answers drag that fit down along with them, so the model stops being
    surprised by them.

    This statistic avoids that by never pooling the two regimes. Split the
    items asked at their median difficulty, estimate ability separately on each
    side, and test whether the two estimates agree:

        z = (theta_easy - theta_hard) / sqrt(se_easy^2 + se_hard^2)

    A genuine annotator has one ability, so the two estimates differ only by
    sampling noise and z is around zero. Someone answering from a key covering
    only the easy end looks strong on one side and like a guesser on the other,
    which is a large positive z no matter where the pooled fit lands.

    The prior is deliberately loosened (sd 2.0 rather than the population 1.0)
    because each side has few items and the tighter prior would shrink both
    estimates toward each other, which is precisely the signal being measured.

    Returns 0.0 when either side is too small to estimate.
    """
    if len(asked) < 2 * min_per_side:
        return 0.0

    idx = np.asarray(asked, dtype=int)
    ok = np.asarray(correct, dtype=bool)
    b = bank.b[idx]
    cut = float(np.median(b))

    easy = b <= cut
    hard = ~easy
    if easy.sum() < min_per_side or hard.sum() < min_per_side:
        return 0.0

    t_easy, se_easy = estimate_ability(
        bank, idx[easy].tolist(), ok[easy].tolist(), prior_sd=prior_sd
    )
    t_hard, se_hard = estimate_ability(
        bank, idx[hard].tolist(), ok[hard].tolist(), prior_sd=prior_sd
    )

    denom = math.sqrt(se_easy**2 + se_hard**2)
    if denom < 1e-9:
        return 0.0
    return (t_easy - t_hard) / denom
