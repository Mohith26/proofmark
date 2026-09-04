import numpy as np

from proofmark.detect import SPRT, detection_information
from proofmark.irt import ItemBank, ability_posterior, marginal_p_correct
from proofmark.router import (
    DetectionRouter,
    FisherRouter,
    HybridRouter,
    RandomRouter,
    ROUTERS,
)


def _dense_bank(n=601, c=0.25):
    b = np.linspace(-4, 4, n)
    return ItemBank(a=np.full(n, 1.5), b=b, c=np.full(n, c))


def test_the_two_objectives_peak_at_different_difficulties():
    """The central claim of the project.

    Fisher information about ability and information about honesty are
    maximised by different gold questions. Fisher wants something near the
    annotator's own level; the honesty test wants something easy, because that
    is where an honest annotator and a chance-level guesser diverge most.
    """
    bank = _dense_bank()
    post = ability_posterior(bank, [], [])

    fisher = bank.fisher_information(0.0)
    detect = np.array(
        [
            detection_information(marginal_p_correct(bank, i, post), float(bank.c[i]), 0.5)
            for i in range(len(bank))
        ]
    )

    b_fisher = float(bank.b[int(np.argmax(fisher))])
    b_detect = float(bank.b[int(np.argmax(detect))])

    # Detection prefers strictly easier items (lower difficulty).
    assert b_detect < b_fisher
    # And the gap is large, not a rounding artefact.
    assert b_fisher - b_detect > 1.0


def test_fisher_router_picks_near_the_current_ability_estimate():
    bank = _dense_bank(c=0.0)
    r = FisherRouter(bank, rng=np.random.default_rng(0))
    post = ability_posterior(bank, [], [])  # centred on 0
    idx = r.select(np.arange(len(bank)), post, SPRT())
    assert abs(bank.b[idx]) < 0.15


def test_detection_router_picks_an_easy_item():
    bank = _dense_bank()
    r = DetectionRouter(bank, rng=np.random.default_rng(0))
    post = ability_posterior(bank, [], [])
    idx = r.select(np.arange(len(bank)), post, SPRT())
    assert bank.b[idx] < -2.0


def test_hybrid_starts_on_detection_and_ends_on_ability():
    bank = _dense_bank()
    post = ability_posterior(bank, [], [])
    r = HybridRouter(bank, rng=np.random.default_rng(0), top_k=1)

    fresh = SPRT()
    early = bank.b[r.select(np.arange(len(bank)), post, fresh)]

    settled = SPRT()
    settled.llr = settled.upper  # honesty question resolved
    late = bank.b[r.select(np.arange(len(bank)), post, settled)]

    assert early < late


def test_routers_never_return_an_unavailable_item():
    bank = ItemBank.synthetic(50, seed=3)
    post = ability_posterior(bank, [], [])
    available = np.array([7, 11, 23, 44])
    for name, cls in ROUTERS.items():
        r = cls(bank, rng=np.random.default_rng(1))
        for _ in range(20):
            idx = r.select(available, post, SPRT())
            assert idx in available, name


def test_router_raises_when_the_pool_is_empty():
    bank = ItemBank.synthetic(5, seed=4)
    post = ability_posterior(bank, [], [])
    r = RandomRouter(bank)
    try:
        r.select(np.array([], dtype=int), post, SPRT())
    except ValueError:
        return
    raise AssertionError("expected ValueError on an empty pool")


def test_top_k_sampling_makes_the_policy_non_deterministic():
    # Predictability is what the gold-savvy adversary exploits, so the hybrid
    # policy deliberately samples among its best few candidates.
    bank = ItemBank.synthetic(200, seed=5)
    post = ability_posterior(bank, [], [])
    available = np.arange(len(bank))

    fixed = HybridRouter(bank, rng=np.random.default_rng(0), top_k=1)
    picks_fixed = {fixed.select(available, post, SPRT()) for _ in range(25)}
    assert len(picks_fixed) == 1

    spread = HybridRouter(bank, rng=np.random.default_rng(0), top_k=5)
    picks_spread = {spread.select(available, post, SPRT()) for _ in range(25)}
    assert len(picks_spread) > 1
