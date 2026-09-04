"""Run a routing policy against a simulated annotator and score the result."""

from dataclasses import dataclass

import numpy as np

from .irt import ability_posterior, marginal_p_correct, THETA_GRID
from .detect import SPRT
from .personfit import difficulty_spread, lz, split_difficulty_z
from .router import ROUTERS

# Threshold on the split-difficulty consistency statistic. It is a ratio of a
# difference to its standard error, so it is roughly standard normal for an
# honest annotator and 2.5 is about the 1-in-160 upper tail. The realised
# false-positive rate is measured in the benchmark rather than assumed, since
# the two ability estimates are not perfectly independent.
SPLIT_Z_THRESHOLD = 2.5

# Kept for the record. lz is the textbook person-fit index and the obvious
# thing to reach for here, but the benchmark shows it has no power against a
# memorised answer key, so nothing depends on it. See personfit.split_difficulty_z.
LZ_THRESHOLD = -3.0


@dataclass
class TrialResult:
    decision: str  # 'honest', 'spam', or 'undecided' -- the sequential test alone
    n_questions: int
    theta_true: float | None
    theta_hat: float
    theta_se: float
    is_dishonest: bool
    lz: float = 0.0
    split_z: float = 0.0
    difficulty_spread: float = 0.0

    def flagged(self, split_z_threshold=None, lz_threshold=None):
        """Combined verdict.

        Two independent failure modes, each with its own detector:
        chance-level responding (the sequential test) and an ability that
        depends on item difficulty (the split-difficulty test).
        """
        if self.decision == "spam":
            return True
        if split_z_threshold is not None and self.split_z > split_z_threshold:
            return True
        if lz_threshold is not None and self.lz < lz_threshold:
            return True
        return False


def run_trial(
    bank,
    annotator,
    is_dishonest,
    router,
    budget=40,
    alpha=0.01,
    beta=0.01,
    min_questions=0,
):
    """Ask gold questions until the sequential test decides or we run out of budget.

    min_questions holds off on accepting an 'honest' verdict until at least
    that many questions have been asked. This exists because of a real
    interaction between the two detectors, and it costs real money, so it
    defaults to off.

    The sequential test is efficient precisely because it stops early. But the
    split-difficulty check in personfit.py needs a reasonable number of items
    on both sides of the difficulty median before it can say anything, and an
    annotator working from a memorised answer key gets cleared by the
    sequential test in about fourteen questions. So the early stop that makes
    the first detector cheap is exactly what blinds the second one. Raising
    this floor buys back the second detector's power at the cost of spending
    more gold on annotators who are fine. The benchmark measures both sides.

    A 'spam' verdict always stops immediately; there is nothing more to learn.
    """
    sprt = SPRT(alpha=alpha, beta=beta)
    available = np.arange(len(bank))
    asked, correct = [], []
    posterior = ability_posterior(bank, asked, correct)

    for _ in range(budget):
        idx = router.select(available, posterior, sprt)

        # p_honest has to be computed from the posterior BEFORE this item's
        # answer is folded in, otherwise the likelihood ratio is conditioned on
        # the very observation it is scoring and the test is badly optimistic.
        p_honest = marginal_p_correct(bank, idx, posterior)
        p_spam = float(bank.c[idx])

        ok = annotator.respond(bank, idx)
        asked.append(idx)
        correct.append(ok)
        available = available[available != idx]

        posterior = ability_posterior(bank, asked, correct)
        decision = sprt.update(ok, p_honest, p_spam)
        if decision == "spam":
            break
        if decision == "honest" and len(asked) >= min_questions:
            break
        if len(available) == 0:
            break

    mean = float(np.sum(THETA_GRID * posterior))
    var = float(np.sum((THETA_GRID - mean) ** 2 * posterior))
    return TrialResult(
        decision=sprt.decision or "undecided",
        n_questions=len(asked),
        theta_true=getattr(annotator, "theta", None),
        theta_hat=mean,
        theta_se=float(np.sqrt(var)),
        is_dishonest=is_dishonest,
        lz=lz(bank, asked, correct, mean),
        split_z=split_difficulty_z(bank, asked, correct),
        difficulty_spread=difficulty_spread(bank, asked),
    )


def summarise(results, split_z_threshold=None, lz_threshold=None):
    """Detection quality plus ability-estimation error, in one dict.

    Pass split_z_threshold to score the combined rule instead of the sequential
    test alone.
    """
    flagged = [r for r in results if r.flagged(split_z_threshold, lz_threshold)]
    dishonest = [r for r in results if r.is_dishonest]

    tp = sum(1 for r in flagged if r.is_dishonest)
    fp = len(flagged) - tp

    precision = tp / len(flagged) if flagged else float("nan")
    recall = tp / len(dishonest) if dishonest else float("nan")
    f1 = (
        2 * precision * recall / (precision + recall)
        if flagged and dishonest and (precision + recall) > 0
        else float("nan")
    )

    honest = [r for r in results if not r.is_dishonest and r.theta_true is not None]
    rmse = (
        float(np.sqrt(np.mean([(r.theta_hat - r.theta_true) ** 2 for r in honest])))
        if honest
        else float("nan")
    )

    # False positive rate on honest annotators is the number that actually
    # matters operationally -- wrongly suspending a good annotator is far more
    # expensive than letting one spammer through for another day.
    honest_all = [r for r in results if not r.is_dishonest]
    fpr = (
        sum(1 for r in honest_all if r.flagged(split_z_threshold, lz_threshold))
        / len(honest_all)
        if honest_all
        else float("nan")
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": fpr,
        "mean_questions": float(np.mean([r.n_questions for r in results])),
        "mean_questions_dishonest": (
            float(np.mean([r.n_questions for r in dishonest])) if dishonest else float("nan")
        ),
        "ability_rmse": rmse,
        "undecided": sum(1 for r in results if r.decision == "undecided") / len(results),
        "mean_difficulty_spread": float(np.mean([r.difficulty_spread for r in results])),
        "n": len(results),
    }


def run_policy(
    bank,
    population,
    policy_name,
    seed=0,
    budget=40,
    alpha=0.01,
    beta=0.01,
    min_questions=0,
    **router_kwargs,
):
    rng = np.random.default_rng(seed)
    router = ROUTERS[policy_name](bank, rng=rng, **router_kwargs)
    out = []
    for annotator, dishonest in population:
        out.append(
            run_trial(
                bank, annotator, dishonest, router, budget, alpha, beta, min_questions
            )
        )
    return out
