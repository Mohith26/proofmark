"""Annotator population models.

Four behaviours, chosen because they are the ones that actually show up on a
crowd platform and because they break different policies:

  honest      answers according to the IRT model at their true ability
  spam        answers uniformly at random, i.e. at the guessing floor
  gold_savvy  has seen part of the gold pool before and answers those items
              correctly; guesses on everything else. This is the adversary
              that punishes a policy for being predictable, because a policy
              that always picks the same easy items is teaching them exactly
              which items to memorise.
  drifter     honest up to a changepoint, then spam. Models someone who starts
              genuinely working and switches to farming throughput.
"""

import numpy as np


class Annotator:
    kind = "base"

    def respond(self, bank, idx):
        raise NotImplementedError


class HonestAnnotator(Annotator):
    kind = "honest"

    def __init__(self, theta, rng):
        self.theta = theta
        self.rng = rng

    def respond(self, bank, idx):
        p = bank.p_correct_item(idx, self.theta)
        return bool(self.rng.random() < p)


class SpamAnnotator(Annotator):
    kind = "spam"

    def __init__(self, rng):
        self.rng = rng
        self.theta = None

    def respond(self, bank, idx):
        return bool(self.rng.random() < bank.c[idx])


class GoldSavvyAnnotator(Annotator):
    """Guesses, except on the gold items whose answers they already know."""

    kind = "gold_savvy"

    def __init__(self, known, rng):
        self.known = set(int(i) for i in known)
        self.rng = rng
        self.theta = None

    def respond(self, bank, idx):
        if int(idx) in self.known:
            return True
        return bool(self.rng.random() < bank.c[idx])


class DriftingAnnotator(Annotator):
    """Honest, then gives up at a fixed question index."""

    kind = "drifter"

    def __init__(self, theta, changepoint, rng):
        self.theta = theta
        self.changepoint = changepoint
        self.rng = rng
        self.seen = 0

    def respond(self, bank, idx):
        self.seen += 1
        if self.seen > self.changepoint:
            return bool(self.rng.random() < bank.c[idx])
        p = bank.p_correct_item(idx, self.theta)
        return bool(self.rng.random() < p)


def make_population(n, bank, rng, spam_rate=0.15, savvy_rate=0.05, drift_rate=0.05):
    """Build a mixed population. Returns a list of (annotator, is_dishonest)."""
    people = []
    n_items = len(bank)
    for _ in range(n):
        u = rng.random()
        if u < spam_rate:
            people.append((SpamAnnotator(rng), True))
        elif u < spam_rate + savvy_rate:
            # They know the 15% of the pool that is easiest, which is what a
            # naive always-pick-the-easiest policy would have shown them.
            easiest = np.argsort(bank.b)[: max(1, int(0.15 * n_items))]
            people.append((GoldSavvyAnnotator(easiest, rng), True))
        elif u < spam_rate + savvy_rate + drift_rate:
            theta = rng.normal(0.0, 1.0)
            cp = int(rng.integers(3, 12))
            people.append((DriftingAnnotator(theta, cp, rng), True))
        else:
            theta = rng.normal(0.0, 1.0)
            people.append((HonestAnnotator(theta, rng), False))
    return people
