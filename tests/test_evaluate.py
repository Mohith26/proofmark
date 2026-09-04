import numpy as np

from proofmark.evaluate import run_policy, run_trial, summarise
from proofmark.irt import ItemBank
from proofmark.router import HybridRouter, RandomRouter
from proofmark.simulate import (
    DriftingAnnotator,
    GoldSavvyAnnotator,
    HonestAnnotator,
    SpamAnnotator,
    make_population,
)


def test_trial_respects_the_budget():
    bank = ItemBank.synthetic(200, seed=1)
    rng = np.random.default_rng(0)
    a = HonestAnnotator(0.0, rng)
    r = RandomRouter(bank, rng=rng)
    res = run_trial(bank, a, False, r, budget=7)
    assert res.n_questions <= 7


def test_trial_never_repeats_a_question():
    # Enforced indirectly: with a bank smaller than the budget the trial must
    # stop when the pool empties rather than looping forever.
    bank = ItemBank.synthetic(6, seed=2)
    rng = np.random.default_rng(0)
    a = HonestAnnotator(0.0, rng)
    r = RandomRouter(bank, rng=rng)
    res = run_trial(bank, a, False, r, budget=100)
    assert res.n_questions <= 6


def test_spammer_is_caught_and_honest_worker_is_cleared():
    bank = ItemBank.synthetic(300, seed=3)
    rng = np.random.default_rng(4)
    r = HybridRouter(bank, rng=rng)

    spam = run_trial(bank, SpamAnnotator(rng), True, r, budget=60)
    assert spam.decision == "spam"

    good = run_trial(bank, HonestAnnotator(1.0, rng), False, r, budget=60)
    assert good.decision == "honest"


def test_gold_savvy_annotator_answers_known_items_correctly():
    bank = ItemBank.synthetic(50, seed=5)
    rng = np.random.default_rng(0)
    a = GoldSavvyAnnotator({3, 4, 5}, rng)
    assert all(a.respond(bank, i) for i in (3, 4, 5))


def test_drifter_switches_behaviour_at_the_changepoint():
    bank = ItemBank(a=np.array([2.0]), b=np.array([-6.0]), c=np.array([0.25]))
    rng = np.random.default_rng(0)
    a = DriftingAnnotator(3.0, changepoint=5, rng=rng)
    # Very easy item plus high ability: an honest answer is essentially certain.
    before = [a.respond(bank, 0) for _ in range(5)]
    assert all(before)
    after = [a.respond(bank, 0) for _ in range(400)]
    # Post-changepoint they are guessing, so roughly a quarter should be right.
    assert 0.15 < sum(after) / len(after) < 0.35


def test_population_has_the_requested_mix():
    bank = ItemBank.synthetic(100, seed=6)
    rng = np.random.default_rng(1)
    pop = make_population(4000, bank, rng, spam_rate=0.2, savvy_rate=0.0, drift_rate=0.0)
    frac = sum(1 for _, d in pop if d) / len(pop)
    assert 0.17 < frac < 0.23


def test_summary_metrics_are_well_formed():
    bank = ItemBank.synthetic(300, seed=7)
    rng = np.random.default_rng(2)
    pop = make_population(60, bank, rng)
    res = run_policy(bank, pop, "hybrid", seed=3, budget=40)
    s = summarise(res)
    for key in ("precision", "recall", "f1", "false_positive_rate", "ability_rmse"):
        assert key in s
    assert 0.0 <= s["false_positive_rate"] <= 1.0
    assert s["n"] == 60
