import math

import numpy as np

from proofmark.detect import SPRT, detection_information, kl_bernoulli, llr_increment
from proofmark.irt import ItemBank


def test_kl_is_zero_only_when_identical():
    assert kl_bernoulli(0.4, 0.4) == 0.0
    assert kl_bernoulli(0.9, 0.25) > 0
    assert kl_bernoulli(0.25, 0.9) > 0


def test_kl_grows_as_the_distributions_separate():
    base = kl_bernoulli(0.5, 0.25)
    more = kl_bernoulli(0.8, 0.25)
    most = kl_bernoulli(0.99, 0.25)
    assert base < more < most


def test_llr_increment_points_the_right_way():
    # Getting a question right is evidence for honesty when the honest model
    # predicts a higher success rate than chance, and against it when wrong.
    assert llr_increment(True, 0.9, 0.25) > 0
    assert llr_increment(False, 0.9, 0.25) < 0


def test_detection_information_vanishes_when_hypotheses_coincide():
    # An item so hard that an honest annotator is also at chance carries no
    # information about whether they are cheating. This is the whole reason
    # adaptive testing is a bad fraud detector.
    assert detection_information(0.25, 0.25, 0.5) == 0.0
    assert detection_information(0.99, 0.25, 0.5) > 0.5


def test_detection_information_rises_with_honest_success_rate():
    vals = [detection_information(p, 0.25, 0.5) for p in (0.3, 0.5, 0.7, 0.9, 0.99)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_sprt_boundaries_match_wald():
    s = SPRT(alpha=0.01, beta=0.02)
    assert abs(s.upper - math.log(0.98 / 0.01)) < 1e-12
    assert abs(s.lower - math.log(0.02 / 0.99)) < 1e-12


def test_sprt_rejects_bad_error_rates():
    for a, b in ((0.0, 0.1), (1.0, 0.1), (0.1, 0.0), (0.1, 1.0)):
        try:
            SPRT(alpha=a, beta=b)
        except ValueError:
            continue
        raise AssertionError(f"should have rejected alpha={a} beta={b}")


def test_sprt_clears_a_consistently_correct_annotator():
    s = SPRT()
    for _ in range(50):
        if s.update(True, 0.9, 0.25) is not None:
            break
    assert s.decision == "honest"


def test_sprt_flags_a_chance_level_annotator():
    rng = np.random.default_rng(0)
    s = SPRT()
    for _ in range(200):
        ok = bool(rng.random() < 0.25)
        if s.update(ok, 0.9, 0.25) is not None:
            break
    assert s.decision == "spam"


def test_resolution_stays_in_unit_interval():
    s = SPRT()
    assert s.resolution == 0.0
    for _ in range(200):
        s.update(True, 0.9, 0.25)
        assert 0.0 <= s.resolution <= 1.0
    assert s.resolution == 1.0


def test_posterior_honest_is_monotone_in_llr():
    s = SPRT()
    prev = s.posterior_honest
    for _ in range(10):
        s.llr += 0.3
        assert s.posterior_honest > prev
        prev = s.posterior_honest


def test_posterior_honest_saturates_without_overflow():
    s = SPRT()
    s.llr = 10_000.0
    assert s.posterior_honest == 1.0
    s.llr = -10_000.0
    assert s.posterior_honest == 0.0
