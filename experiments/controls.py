"""Negative controls.

Three places where this system could quietly be wrong. Rather than assert in
the README that I handled them, each one is reintroduced deliberately here and
the damage is measured.

    python experiments/controls.py
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proofmark.detect import SPRT
from proofmark.irt import ItemBank, ability_posterior, marginal_p_correct
from proofmark.router import HybridRouter
from proofmark.simulate import HonestAnnotator, SpamAnnotator


def _trial(bank, annotator, router, budget, leak_future):
    """One trial. leak_future=True reintroduces the conditioning bug."""
    sprt = SPRT()
    available = np.arange(len(bank))
    asked, correct = [], []
    posterior = ability_posterior(bank, asked, correct)

    for _ in range(budget):
        idx = router.select(available, posterior, sprt)
        ok = annotator.respond(bank, idx)

        asked.append(idx)
        correct.append(ok)
        available = available[available != idx]
        new_posterior = ability_posterior(bank, asked, correct)

        # The bug: computing P(correct | honest) from a posterior that has
        # already absorbed this very answer. The honest model then explains the
        # observation almost perfectly no matter what was observed, the
        # likelihood ratio stops discriminating, and spammers look honest.
        p_honest = (
            marginal_p_correct(bank, idx, new_posterior)
            if leak_future
            else marginal_p_correct(bank, idx, posterior)
        )
        posterior = new_posterior

        if sprt.update(ok, p_honest, float(bank.c[idx])) is not None:
            break
        if len(available) == 0:
            break

    return sprt.decision or "undecided", len(asked)


def control_posterior_leak(n=500, seed=3):
    """Control 1: condition the likelihood ratio on the answer being scored."""
    bank = ItemBank.synthetic(400, seed=seed)
    out = {}
    for leak in (False, True):
        rng = np.random.default_rng(seed + 1)
        router = HybridRouter(bank, rng=rng)
        caught = 0
        for _ in range(n):
            d, _ = _trial(bank, SpamAnnotator(rng), router, 40, leak)
            caught += d == "spam"
        out["leaked" if leak else "correct"] = round(caught / n, 3)
    out["recall_lost"] = round(out["correct"] - out["leaked"], 3)
    return out


def control_nominal_vs_realised_error(n=3000, seed=7):
    """Control 2: how far the composite hypothesis pushes us off Wald's bounds.

    Wald's boundaries are exact for two simple hypotheses. Our honest
    hypothesis is composite because ability is unknown and we marginalise over
    a posterior that is itself being learned. So the realised false-positive
    rate is not the nominal alpha. This measures the gap instead of assuming
    it away.
    """
    bank = ItemBank.synthetic(400, seed=seed)
    out = {}
    for alpha in (0.05, 0.01, 0.001):
        rng = np.random.default_rng(seed + 2)
        router = HybridRouter(bank, rng=rng)
        false_flags = 0
        for _ in range(n):
            theta = rng.normal(0.0, 1.0)
            sprt = SPRT(alpha=alpha, beta=alpha)
            available = np.arange(len(bank))
            asked, correct = [], []
            posterior = ability_posterior(bank, asked, correct)
            a = HonestAnnotator(theta, rng)
            for _ in range(40):
                idx = router.select(available, posterior, sprt)
                p_honest = marginal_p_correct(bank, idx, posterior)
                ok = a.respond(bank, idx)
                asked.append(idx)
                correct.append(ok)
                available = available[available != idx]
                posterior = ability_posterior(bank, asked, correct)
                if sprt.update(ok, p_honest, float(bank.c[idx])) is not None:
                    break
            false_flags += sprt.decision == "spam"
        out[f"alpha={alpha}"] = {
            "nominal": alpha,
            "realised_false_positive_rate": round(false_flags / n, 4),
            "ratio": round((false_flags / n) / alpha, 2) if alpha else None,
        }
    return out


def control_mean_vs_marginal(n=400, seed=13):
    """Control 3: plug in the point ability estimate instead of marginalising.

    Cheaper and tempting. It also throws away the uncertainty in the ability
    estimate, which is largest exactly when we have asked the fewest questions,
    which is exactly when the sequential test is most fragile.
    """
    from proofmark.irt import THETA_GRID

    bank = ItemBank.synthetic(400, seed=seed)
    out = {}
    for marginalise in (True, False):
        rng = np.random.default_rng(seed + 3)
        router = HybridRouter(bank, rng=rng)
        false_flags, honest_n = 0, 0
        caught, spam_n = 0, 0
        for i in range(n * 2):
            is_spam = i % 2 == 0
            a = SpamAnnotator(rng) if is_spam else HonestAnnotator(rng.normal(0, 1), rng)
            sprt = SPRT()
            available = np.arange(len(bank))
            asked, correct = [], []
            posterior = ability_posterior(bank, asked, correct)
            for _ in range(40):
                idx = router.select(available, posterior, sprt)
                if marginalise:
                    p_honest = marginal_p_correct(bank, idx, posterior)
                else:
                    theta_hat = float(THETA_GRID @ posterior)
                    p_honest = bank.p_correct_item(idx, theta_hat)
                ok = a.respond(bank, idx)
                asked.append(idx)
                correct.append(ok)
                available = available[available != idx]
                posterior = ability_posterior(bank, asked, correct)
                if sprt.update(ok, p_honest, float(bank.c[idx])) is not None:
                    break
            if is_spam:
                spam_n += 1
                caught += sprt.decision == "spam"
            else:
                honest_n += 1
                false_flags += sprt.decision == "spam"
        out["marginalised" if marginalise else "point_estimate"] = {
            "recall": round(caught / spam_n, 3),
            "false_positive_rate": round(false_flags / honest_n, 4),
        }
    return out


def main():
    results = {}

    print("=" * 70)
    print("Control 1: score the answer using a posterior that already saw it")
    print("=" * 70)
    c1 = control_posterior_leak()
    results["posterior_leak"] = c1
    print(f"  recall on spammers, computed correctly : {c1['correct']:.3f}")
    print(f"  recall with the leaked posterior        : {c1['leaked']:.3f}")
    print(f"  recall lost to the bug                  : {c1['recall_lost']:.3f}")

    print()
    print("=" * 70)
    print("Control 2: nominal alpha vs the false-positive rate we actually get")
    print("=" * 70)
    c2 = control_nominal_vs_realised_error()
    results["nominal_vs_realised"] = c2
    print(f"{'nominal alpha':>15}{'realised FPR':>16}{'ratio':>10}")
    print("-" * 41)
    for k, v in c2.items():
        print(f"{v['nominal']:>15}{v['realised_false_positive_rate']:>16.4f}{v['ratio']:>10}")

    print()
    print("=" * 70)
    print("Control 3: point ability estimate instead of marginalising")
    print("=" * 70)
    c3 = control_mean_vs_marginal()
    results["mean_vs_marginal"] = c3
    print(f"{'variant':<18}{'recall':>10}{'FPR':>10}")
    print("-" * 38)
    for k, v in c3.items():
        print(f"{k:<18}{v['recall']:>10.3f}{v['false_positive_rate']:>10.4f}")

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "controls.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote experiments/controls.json")


if __name__ == "__main__":
    main()
