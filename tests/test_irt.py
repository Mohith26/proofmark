import numpy as np

from proofmark.irt import (
    ItemBank,
    THETA_GRID,
    ability_posterior,
    estimate_ability,
    marginal_p_correct,
    sigmoid,
)


def test_sigmoid_is_stable_at_extremes():
    z = np.array([-800.0, -1.0, 0.0, 1.0, 800.0])
    s = sigmoid(z)
    assert np.all(np.isfinite(s))
    assert s[0] == 0.0
    assert abs(s[2] - 0.5) < 1e-12
    assert s[4] == 1.0


def test_p_correct_is_monotone_in_ability():
    bank = ItemBank.synthetic(30, seed=1)
    lo = bank.p_correct(-2.0)
    mid = bank.p_correct(0.0)
    hi = bank.p_correct(2.0)
    assert np.all(lo < mid)
    assert np.all(mid < hi)


def test_p_correct_bottoms_out_at_the_guessing_floor():
    bank = ItemBank.synthetic(20, n_choices=4, seed=2)
    p = bank.p_correct(-50.0)
    assert np.allclose(p, 0.25, atol=1e-6)
    p_hi = bank.p_correct(50.0)
    assert np.allclose(p_hi, 1.0, atol=1e-6)


def test_bank_rejects_bad_parameters():
    for kwargs in (
        dict(a=np.array([1.0]), b=np.array([0.0]), c=np.array([0.0, 0.0])),
        dict(a=np.array([-1.0]), b=np.array([0.0]), c=np.array([0.0])),
        dict(a=np.array([1.0]), b=np.array([0.0]), c=np.array([1.0])),
    ):
        try:
            ItemBank(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"should have rejected {kwargs}")


def test_fisher_information_peaks_at_matched_difficulty_when_no_guessing():
    # With c = 0 the 3PL collapses to 2PL and the peak is exactly at b = theta.
    b = np.linspace(-3, 3, 601)
    bank = ItemBank(a=np.ones_like(b), b=b, c=np.zeros_like(b))
    info = bank.fisher_information(0.0)
    assert abs(b[int(np.argmax(info))]) < 0.02


def test_posterior_concentrates_as_evidence_accumulates():
    bank = ItemBank.synthetic(200, seed=3)
    rng = np.random.default_rng(0)
    theta = 1.0
    asked, correct = [], []
    widths = []
    for n, idx in enumerate(range(60)):
        p = bank.p_correct_item(idx, theta)
        asked.append(idx)
        correct.append(bool(rng.random() < p))
        if n in (9, 29, 59):
            _, sd = estimate_ability(bank, asked, correct)
            widths.append(sd)
    assert widths[0] > widths[1] > widths[2]


def test_ability_estimate_recovers_truth():
    bank = ItemBank.synthetic(400, seed=4)
    rng = np.random.default_rng(7)
    errs = []
    for theta in (-1.5, -0.5, 0.5, 1.5):
        asked = list(range(400))
        correct = [bool(rng.random() < bank.p_correct_item(i, theta)) for i in asked]
        hat, _ = estimate_ability(bank, asked, correct)
        errs.append(abs(hat - theta))
    # Shrinkage toward the prior mean keeps this from being exact, but 400
    # items should land every estimate well inside half a logit.
    assert max(errs) < 0.5


def test_posterior_sums_to_one():
    bank = ItemBank.synthetic(10, seed=5)
    post = ability_posterior(bank, [0, 1, 2], [True, False, True])
    assert abs(post.sum() - 1.0) < 1e-12
    assert len(post) == len(THETA_GRID)


def test_marginal_p_correct_sits_between_floor_and_one():
    bank = ItemBank.synthetic(10, seed=6)
    post = ability_posterior(bank, [], [])
    for idx in range(10):
        p = marginal_p_correct(bank, idx, post)
        assert bank.c[idx] - 1e-9 <= p <= 1.0


def test_marginal_p_correct_tracks_the_posterior():
    # A run of correct answers should raise the marginal probability that the
    # annotator gets the next item right.
    bank = ItemBank.synthetic(60, seed=8)
    flat = ability_posterior(bank, [], [])
    strong = ability_posterior(bank, list(range(20)), [True] * 20)
    assert marginal_p_correct(bank, 55, strong) > marginal_p_correct(bank, 55, flat)
