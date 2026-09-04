"""Sequential test for 'is this annotator actually working?'.

We run Wald's sequential probability ratio test on each annotator's stream of
gold-question outcomes. Two hypotheses:

    H_honest: responses follow the IRT model at the annotator's ability
    H_spam:   responses are at the guessing floor c, regardless of difficulty

The SPRT is the right tool because it stops as soon as it has enough evidence
rather than after a fixed number of questions, and gold questions cost real
money. The quantity that decides how fast it stops is the expected drift of the
log-likelihood ratio, which is a KL divergence, and that is what the detection
half of the router maximises.
"""

import math

# Guard rails on probabilities so no log blows up on a degenerate item.
_EPS = 1e-9


def _clip(p):
    return min(max(p, _EPS), 1.0 - _EPS)


def llr_increment(correct, p_honest, p_spam):
    """One term of the log-likelihood ratio, honest over spam."""
    ph, ps = _clip(p_honest), _clip(p_spam)
    if correct:
        return math.log(ph / ps)
    return math.log((1.0 - ph) / (1.0 - ps))


def kl_bernoulli(p, q):
    """KL(Bern(p) || Bern(q)) in nats."""
    p, q = _clip(p), _clip(q)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def detection_information(p_honest, p_spam, posterior_honest):
    """Expected magnitude of the LLR step this item would produce.

    Under H_honest the LLR drifts up by KL(honest || spam) per question; under
    H_spam it drifts down by KL(spam || honest). Weighting the two by our
    current belief gives the expected absolute drift, which is the thing we
    actually want to maximise if the goal is to resolve the question quickly in
    whichever direction the truth lies.

    Note this goes to zero as p_honest approaches p_spam, which is what happens
    on very hard items where an honest annotator is also guessing. Hard items
    are informative about ability and worthless for detection.
    """
    fwd = kl_bernoulli(p_honest, p_spam)
    rev = kl_bernoulli(p_spam, p_honest)
    return posterior_honest * fwd + (1.0 - posterior_honest) * rev


class SPRT:
    """Wald's SPRT with the usual approximate boundaries.

    alpha is the tolerated probability of flagging an honest annotator, beta
    the tolerated probability of clearing a spammer. Wald's bounds

        upper = log((1 - beta) / alpha)
        lower = log(beta / (1 - alpha))

    are exact only for simple hypotheses. Ours is composite on the honest side
    because ability is unknown, so the realised error rates drift away from the
    nominal ones. controls.py measures how far.
    """

    def __init__(self, alpha=0.01, beta=0.01):
        if not (0 < alpha < 1 and 0 < beta < 1):
            raise ValueError("alpha and beta must be in (0, 1)")
        self.alpha = alpha
        self.beta = beta
        self.upper = math.log((1.0 - beta) / alpha)
        self.lower = math.log(beta / (1.0 - alpha))
        self.llr = 0.0
        self.n = 0
        self.decision = None  # None while undecided, else 'honest' or 'spam'

    def update(self, correct, p_honest, p_spam):
        self.llr += llr_increment(correct, p_honest, p_spam)
        self.n += 1
        if self.llr >= self.upper:
            self.decision = "honest"
        elif self.llr <= self.lower:
            self.decision = "spam"
        return self.decision

    @property
    def posterior_honest(self):
        """Posterior P(honest) from the LLR under a flat prior over the two."""
        if self.llr > 40:
            return 1.0
        if self.llr < -40:
            return 0.0
        return 1.0 / (1.0 + math.exp(-self.llr))

    @property
    def resolution(self):
        """How far the test has travelled toward a decision, in [0, 1].

        0 means we know nothing about honesty, 1 means a boundary is reached.
        The router uses this to decide how much effort to keep spending on
        detection versus ability measurement.
        """
        if self.llr >= 0:
            return min(1.0, self.llr / self.upper)
        return min(1.0, self.llr / self.lower)
