"""Main results.

    python experiments/run_benchmark.py

Writes results.json next to this file and prints the tables that the README
quotes. Everything is seeded, so the numbers reproduce exactly.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proofmark.detect import detection_information
from proofmark.irt import ItemBank, ability_posterior, marginal_p_correct_all
from proofmark.evaluate import (
    LZ_THRESHOLD,
    SPLIT_Z_THRESHOLD,
    run_policy,
    run_trial,
    summarise,
)
from proofmark.simulate import GoldSavvyAnnotator, HonestAnnotator, make_population
from proofmark.router import HybridRouter

POLICIES = ["random", "fisher", "detection", "hybrid"]


def objective_peaks():
    """Where each objective is maximised, on a dense difficulty sweep.

    This is the observation the whole project is built on. Both objectives are
    evaluated for an annotator we know nothing about yet (flat prior, ability
    estimate 0), across a bank whose only varying parameter is difficulty.
    """
    n = 1601
    b = np.linspace(-5, 5, n)
    bank = ItemBank(a=np.full(n, 1.5), b=b, c=np.full(n, 0.25))
    post = ability_posterior(bank, [], [])

    fisher = bank.fisher_information(0.0)
    ph = marginal_p_correct_all(bank, post)
    detect = np.array(
        [detection_information(ph[i], 0.25, 0.5) for i in range(n)]
    )

    i_f, i_d = int(np.argmax(fisher)), int(np.argmax(detect))

    # Fisher information has a genuine interior maximum. Detection information
    # does not -- it increases monotonically as items get easier and simply
    # runs into the edge of whatever range you sweep. Reporting a "peak
    # difficulty" for it would be an artefact of the sweep bounds, so the
    # honest summary is the monotonicity plus how much of each objective the
    # other one's preferred item throws away.
    detect_is_monotone = bool(np.all(np.diff(detect) <= 1e-12))

    return {
        "fisher_peak_difficulty": round(float(b[i_f]), 3),
        "fisher_peak_is_interior": bool(0 < i_f < n - 1),
        "p_correct_at_fisher_peak": round(float(ph[i_f]), 3),
        "detection_is_monotone_in_easiness": detect_is_monotone,
        "detection_argmax_at_sweep_edge": bool(i_d in (0, n - 1)),
        "p_correct_at_detection_argmax": round(float(ph[i_d]), 3),
        # The numbers that actually matter: what each objective's preferred
        # item costs you on the other objective.
        "detection_info_retained_by_fisher_pick": round(
            float(detect[i_f] / detect[i_d]), 4
        ),
        "fisher_info_retained_by_detection_pick": round(
            float(fisher[i_d] / fisher[i_f]), 4
        ),
    }


def policy_comparison(n_annotators=800, n_items=400, budget=40, seed=11):
    bank = ItemBank.synthetic(n_items, seed=seed)
    rng = np.random.default_rng(seed)
    population = make_population(n_annotators, bank, rng)

    rows = {}
    for name in POLICIES:
        t0 = time.time()
        res = run_policy(bank, population, name, seed=seed + 1, budget=budget)
        s = summarise(res)
        s["seconds"] = round(time.time() - t0, 2)
        rows[name] = s
    return rows


def gold_leak_stress(n_items=400, n_trials=400, seed=23):
    """An annotator working from a memorised answer key for the easiest 15%.

    Two things get measured here and the first one is a negative result.

    Sampling among the top-k scoring items instead of the single best sounds
    like the obvious defence against a predictable policy. It does nothing,
    because the top k detection-scored items all live in the same easy corner
    of the bank that the adversary memorised. The sequential test also cannot
    help: getting easy items right and hard items wrong is not chance
    responding, it is what an ordinary mediocre annotator looks like.

    Two more things that sound right and are not, both measured below.

    lz, the textbook person-fit index, has no power at all -- and its sign is
    backwards. A memorised answer key produces a pattern that is *more*
    orderly than the model predicts, not less, so lz goes positive. lz is
    built to catch careless or random responding, not responding that is too
    clean.

    Spending part of the budget on uniformly random items, to force difficulty
    spread into the record, actively makes detection worse. Random items land
    in territory where honest annotators are guessing too, which washes out
    exactly the easy-versus-hard contrast the detector runs on.

    What does work is comparing the ability estimated from the easy half of
    the items asked against the ability estimated from the hard half.
    """
    bank = ItemBank.synthetic(n_items, seed=seed)
    easiest = np.argsort(bank.b)[: max(1, int(0.15 * n_items))]

    settings = [
        (3, 0.0, 0),
        (1, 0.0, 40),
        (3, 0.0, 40),
        (10, 0.0, 40),
        (3, 0.0, 20),
        (3, 0.15, 40),
        (3, 0.3, 40),
        (3, 0.5, 40),
    ]

    out = {}
    for top_k, eps, min_q in settings:
        rng = np.random.default_rng(seed + 5)
        router = HybridRouter(bank, rng=rng, top_k=top_k, epsilon=eps)

        sprt_only, with_lz, with_split, spread = 0, 0, 0, []
        for _ in range(n_trials):
            r = run_trial(
                bank, GoldSavvyAnnotator(easiest, rng), True, router,
                budget=40, min_questions=min_q,
            )
            sprt_only += r.decision == "spam"
            with_lz += r.flagged(lz_threshold=LZ_THRESHOLD)
            with_split += r.flagged(split_z_threshold=SPLIT_Z_THRESHOLD)
            spread.append(r.difficulty_spread)

        # The same settings have to be checked against honest annotators, or
        # "recall" here means nothing: flagging everyone would score 1.00.
        # The gold spent on those honest annotators is what the extra recall
        # actually costs.
        honest_flagged, honest_cost = 0, []
        for _ in range(n_trials):
            h = run_trial(
                bank, HonestAnnotator(rng.normal(0, 1), rng), False, router,
                budget=40, min_questions=min_q,
            )
            honest_flagged += h.flagged(split_z_threshold=SPLIT_Z_THRESHOLD)
            honest_cost.append(h.n_questions)

        out[f"k={top_k} eps={eps} min={min_q}"] = {
            "recall_sequential_only": round(sprt_only / n_trials, 3),
            "recall_with_lz": round(with_lz / n_trials, 3),
            "recall_with_split_difficulty": round(with_split / n_trials, 3),
            "false_positive_rate": round(honest_flagged / n_trials, 4),
            "gold_per_honest_worker": round(float(np.mean(honest_cost)), 2),
            "mean_difficulty_spread": round(float(np.mean(spread)), 3),
        }
    return out


def budget_sweep(n_annotators=400, n_items=400, seed=31):
    """Detection recall and ability error as a function of how much gold you buy."""
    bank = ItemBank.synthetic(n_items, seed=seed)
    rng = np.random.default_rng(seed)
    population = make_population(n_annotators, bank, rng)

    sweep = {}
    for budget in (5, 10, 20, 40):
        sweep[budget] = {}
        for name in POLICIES:
            res = run_policy(bank, population, name, seed=seed + 2, budget=budget)
            s = summarise(res)
            sweep[budget][name] = {
                "recall": round(s["recall"], 3),
                "false_positive_rate": round(s["false_positive_rate"], 4),
                "ability_rmse": round(s["ability_rmse"], 3),
            }
    return sweep


def _fmt(x, nd=3):
    if x != x:  # NaN
        return "  n/a"
    return f"{x:.{nd}f}"


def main():
    results = {}

    print("=" * 74)
    print("1. Where each objective peaks (annotator ability 0, 4-way items)")
    print("=" * 74)
    peaks = objective_peaks()
    results["objective_peaks"] = peaks
    print(f"  Fisher information peaks at difficulty {peaks['fisher_peak_difficulty']:+.2f}"
          f" (annotator gets it right {peaks['p_correct_at_fisher_peak']:.0%} of the time),"
          f" interior maximum: {peaks['fisher_peak_is_interior']}")
    print(f"  Detection information is monotone in easiness:"
          f" {peaks['detection_is_monotone_in_easiness']}"
          f" -- it has no interior maximum, it just runs to the easy edge"
          f" (P(correct) = {peaks['p_correct_at_detection_argmax']:.0%} there).")
    print(f"  The Fisher-optimal item carries {peaks['detection_info_retained_by_fisher_pick']:.1%}"
          f" of the available detection information.")
    print(f"  The detection-optimal item carries {peaks['fisher_info_retained_by_detection_pick']:.2%}"
          f" of the available Fisher information.")

    print()
    print("=" * 74)
    print("2. Policy comparison, 800 annotators, 400-item bank, budget 40")
    print("=" * 74)
    rows = policy_comparison()
    results["policy_comparison"] = rows
    hdr = f"{'policy':<11}{'recall':>8}{'FPR':>9}{'F1':>8}{'gold/worker':>13}{'gold/spam':>11}{'ability RMSE':>14}"
    print(hdr)
    print("-" * len(hdr))
    for name in POLICIES:
        s = rows[name]
        print(f"{name:<11}{_fmt(s['recall']):>8}{_fmt(s['false_positive_rate'],4):>9}"
              f"{_fmt(s['f1']):>8}{_fmt(s['mean_questions'],2):>13}"
              f"{_fmt(s['mean_questions_dishonest'],2):>11}{_fmt(s['ability_rmse']):>14}")

    print()
    print("=" * 74)
    print("3. Gold-leak stress: annotator has memorised the easiest 15% of the bank")
    print("=" * 74)
    leak = gold_leak_stress()
    results["gold_leak"] = leak
    hdr3 = (f"{'setting':<22}{'SPRT':>7}{'+lz':>7}{'+split-z':>10}"
            f"{'FPR':>9}{'gold/honest':>13}{'spread':>9}")
    print(hdr3)
    print("-" * len(hdr3))
    for k, v in leak.items():
        print(f"{k:<22}{v['recall_sequential_only']:>7.3f}{v['recall_with_lz']:>7.3f}"
              f"{v['recall_with_split_difficulty']:>10.3f}"
              f"{v['false_positive_rate']:>9.4f}{v['gold_per_honest_worker']:>13.2f}"
              f"{v['mean_difficulty_spread']:>9.3f}")

    print()
    print("=" * 74)
    print("4. Budget sweep")
    print("=" * 74)
    sweep = budget_sweep()
    results["budget_sweep"] = sweep
    print(f"{'budget':<9}{'policy':<12}{'recall':>9}{'FPR':>9}{'ability RMSE':>14}")
    print("-" * 53)
    for budget, per in sweep.items():
        for name in POLICIES:
            v = per[name]
            print(f"{budget:<9}{name:<12}{_fmt(v['recall']):>9}"
                  f"{_fmt(v['false_positive_rate'],4):>9}{_fmt(v['ability_rmse']):>14}")

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote experiments/results.json")


if __name__ == "__main__":
    main()
