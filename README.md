# Proofmark

Adaptive gold-question routing for crowd annotation quality control.

Annotation platforms check whether their workers are actually working by mixing
in "gold questions", tasks whose answer is already known. Gold questions cost
real money, so which one you show next matters. This is a simulator and a set
of routing policies for making that choice well, plus a benchmark that measures
what each policy actually buys you.

The thing that got me interested is that the two obvious goals for a gold
question are in direct conflict, and the conflict is not a tuning problem, it
falls out of the math.

## The conflict

Model the chance an annotator gets a gold question right with a 3PL IRT model,
the standard psychometric one:

```
P(correct | theta) = c + (1 - c) * sigmoid(a * (theta - b))
```

`theta` is the annotator's skill, `b` the item's difficulty, `c` the guessing
floor (1/4 for a 4-way label, because someone who knows nothing still gets a
quarter of them right).

If you want to **measure skill accurately**, you maximise Fisher information,
which peaks on items the annotator gets right about two thirds of the time.
This is what computerised adaptive testing does and it is the right answer to
that question.

If you want to **find out whether they are cheating**, you want the item where
an honest annotator and a chance-level guesser look most different. That is a
KL divergence, and it does not peak anywhere. It climbs monotonically as items
get easier and runs off the easy end of the bank.

Measured on a 1601-item difficulty sweep (`experiments/run_benchmark.py`):

| | best difficulty | annotator gets it right |
|---|---|---|
| Fisher information (measure skill) | -0.21, interior maximum | 67% |
| Detection information (catch cheating) | monotone, runs to the easy edge | 100% |

The cost of picking one over the other:

- The Fisher-optimal item carries **12.8%** of the detection information available.
- The detection-optimal item carries **0.27%** of the Fisher information available.

So a textbook adaptive test spends its entire budget in the region where
cheating is nearly invisible, and vice versa. `HybridRouter` splits the
difference using the sequential test's own progress as the mixing weight: while
the honesty question is open it routes for detection, and once the test is
close to a verdict it switches to measuring skill. Nothing is hand-tuned.

## Results

800 simulated annotators, 400-item bank, 40-question budget, 15% spammers plus
smaller populations of drifters and answer-key users.

| policy | recall | FPR | F1 | gold/worker | gold/spammer | ability RMSE |
|---|---|---|---|---|---|---|
| random | 0.697 | 0.005 | 0.815 | 13.61 | 18.54 | 0.735 |
| fisher (adaptive testing) | 0.808 | 0.008 | 0.882 | 12.43 | 14.10 | **0.481** |
| detection | 0.827 | 0.005 | 0.898 | **4.15** | **3.39** | 0.963 |
| hybrid | **0.832** | 0.008 | 0.896 | 6.61 | 6.12 | 0.710 |

The detection router catches spammers in 3.4 gold questions where adaptive
testing needs 14.1, a **4.2x** reduction, but its skill estimates are twice as
noisy. The hybrid gets the best recall of the four while spending less than
half the gold that adaptive testing does, and recovers most of the skill
accuracy.

At a 5-question budget the gap is starker: the adaptive-testing router catches
**0%** of dishonest annotators, detection-first catches **77%**.

## Three things I expected to work that did not

This is most of what I actually learned, so it is in the README rather than
buried.

I added an adversary who has seen part of the gold pool before: they know the
answers to the easiest 15% of the bank and guess on everything else. The
sequential test catches **0.00** of them, at every setting. That is structural,
not a bug. Answering easy items correctly and hard ones at chance is not
chance-level responding, it is what an ordinary mediocre annotator looks like,
and a test built on aggregate accuracy has nothing to work with.

**1. Randomising which item you pick does not help.** Sampling among the top-k
scoring items instead of the argmax seemed like the obvious defence against a
predictable policy. It does nothing (0.107 at k=1 versus 0.100 at k=3 and 0.065
at k=10), because the top k detection-scored items all live in the same easy
corner of the bank that the adversary memorised in the first place.

**2. The textbook person-fit statistic has no power, and its sign is
backwards.** `lz` (Drasgow, Levine and Williams, 1985) is the standard tool for
detecting response patterns that do not fit the model, so I implemented it. It
catches **0.00** of these annotators. Worse, honest annotators average lz +0.26
and the answer-key users average **+1.24**. lz goes negative for *noisy*
responding; a memorised answer key produces a pattern that is more orderly than
the model predicts, not less. It is the wrong tool and the experiment says so.

**3. Forcing difficulty spread by exploring randomly makes things worse.**
Spending a fraction of the budget on uniformly random items raises the spread
of difficulties asked from 0.64 to 1.64, which is what I wanted, and drives
recall from 0.708 down to **0.025**. Random items land in territory where
honest annotators are guessing too, which washes out the easy-versus-hard
contrast the detector actually runs on.

## What did work

Compare the skill estimated from the easy half of the items asked against the
skill estimated from the hard half, and test whether they agree:

```
z = (theta_easy - theta_hard) / sqrt(se_easy^2 + se_hard^2)
```

A real annotator has one skill level, so the two estimates differ only by
sampling noise. Someone working from a key that covers the easy end looks
strong on one side and like a guesser on the other. This sidesteps the problem
that sank lz, which is that pooling the two regimes lets the aberrant answers
drag the fitted skill down with them until the model is no longer surprised.

It only works if you keep asking. The sequential test clears an answer-key user
in about 14 questions, and the split test needs items on both sides of the
difficulty median before it can say anything, so the early stopping that makes
the first detector cheap is exactly what blinds the second one:

| minimum questions before accepting "honest" | recall | FPR | gold per honest worker |
|---|---|---|---|
| 0 (stop as soon as cleared) | 0.100 | 0.013 | 6.48 |
| 20 | 0.302 | 0.010 | 20.04 |
| 40 (full budget) | **0.708** | 0.013 | 39.64 |

That is the real trade. Catching answer-key users costs about 33 extra gold
questions per honest annotator, and whether that is worth paying depends on
what a gold question costs you versus what a bad annotator costs you. The code
makes it a parameter (`min_questions`) rather than a decision.

## Negative controls

Rather than claim in prose that I handled the tricky parts, `experiments/controls.py`
reintroduces each mistake and measures the damage.

**Scoring an answer with a posterior that already saw it.** The likelihood
ratio needs `P(correct | honest)` computed *before* the answer is folded in.
Using the updated posterior makes the honest model explain every observation
almost perfectly and the ratio stops discriminating. Recall on spammers drops
from 0.996 to 0.860, so the bug costs **13.6 points of recall** while still
looking like it works.

**Wald's boundaries on a composite hypothesis.** The honest hypothesis is
composite (skill is unknown and we marginalise over a posterior we are still
learning), so the nominal error rates are not the real ones. Measured false
positive rate comes in at **0.36 to 0.67 times** the nominal alpha across three
settings, so the test is conservative rather than liberal. Good to know which
direction it errs, and not something I would have wanted to guess.

**Using a point skill estimate instead of marginalising.** Cheaper and
tempting. Recall is unchanged (0.990 vs 0.988) but the false positive rate goes
from 0.010 to 0.025, **2.5x worse**. Marginalising earns its place on false
positives, not on recall, which is not what I assumed going in.

## Running it

```
pip install numpy
python run_tests.py                        # 36 tests
python experiments/run_benchmark.py        # tables above, writes results.json
python experiments/controls.py             # negative controls, writes controls.json
```

Everything is seeded and reproduces exactly. `pytest` works on `tests/` too if
you have it; `run_tests.py` is there so numpy is the only dependency.

## Layout

```
proofmark/
  irt.py         3PL model, ability posteriors, Fisher information
  detect.py      Wald SPRT for chance-level responding, KL drift
  personfit.py   lz (does not work here) and the split-difficulty test (does)
  router.py      random / fisher / detection / hybrid selection policies
  simulate.py    honest, spam, answer-key and drifting annotators
  evaluate.py    trial loop, combined verdict, metrics
tests/           36 tests, no pytest required
experiments/     benchmark and negative controls
```

## References

- Wald (1945), *Sequential Tests of Statistical Hypotheses*
- Birnbaum (1968), latent trait models and Fisher information
- Drasgow, Levine and Williams (1985), *Appropriateness measurement with polychotomous item response models*, for lz
- Dawid and Skene (1979), for the annotator-quality problem this sits next to
