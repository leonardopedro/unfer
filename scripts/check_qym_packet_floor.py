"""Exact finite-window mass check for the decimal three-profile QYM fixture at g=2.

The unnormalized inner occupation basis has diagonal Gram weights prod(n!).
Apply all ladder factors before restricting to total occupation <= 3.
"""
from collections import defaultdict
from fractions import Fraction
from itertools import product
from math import factorial, prod
import json
from pathlib import Path


def words(record):
    """Return (coefficient, ladder word) for the real one-outer-particle matrix."""
    action = record["packet_action"]
    g = 2
    for mode, (metric, curl) in enumerate(
        zip(action["profile_metric"], action["curl_norm_squared"], strict=True)
    ):
        # p = i(a†-a)/2. Store the sign on the annihilation leg of d = a†-a.
        yield -1 / (8 * metric), [(mode, -1)] * 2
        yield curl / 2, [(mode, 1)] * 2
    yield g * Fraction(record["entry_decimal"]), [(i, 1) for i in range(3)]
    for mode, wall in enumerate(action["wall_12_13"], 1):
        yield g*g * wall / 2, [(0, 1)] * 2 + [(mode, 1)] * 2


def act(word, label):
    state = {label: 1}
    for mode, sign in reversed(word):
        result = defaultdict(int)
        for indices, value in state.items():
            n = indices[mode]
            raised = list(indices)
            raised[mode] += 1
            result[tuple(raised)] += value
            if n:
                lowered = list(indices)
                lowered[mode] -= 1
                result[tuple(lowered)] += sign * n * value
        state = result
    return state


def negative_pivots(form, weights, threshold):
    """Exact symmetric elimination of Gram*(H-threshold I), with nonzero pivots."""
    work = [row.copy() for row in form]
    for i, weight in enumerate(weights):
        work[i][i] -= threshold * weight
    count = 0
    for j in range(len(work)):
        pivot = work[j][j]
        assert pivot, "zero pivot requires a different elimination order"
        count += pivot < 0
        for i in range(j + 1, len(work)):
            factor = work[i][j] / pivot
            if factor:
                for k in range(j + 1, len(work)):
                    work[i][k] -= factor * work[j][k]
    return count


def main():
    fixture = Path(__file__).resolve().parents[1] / "docs/fixtures/qym_magnetic_packet.json"
    record = json.loads(fixture.read_text(), parse_float=Fraction)
    assert record["schema"] == "qym-cubic-continuous-packet/v1"
    assert record["coupling"] == 1  # bare cubic coefficient in the fixture
    labels = [n for n in product(range(4), repeat=3) if sum(n) <= 3]
    weights = [prod(factorial(k) for k in n) for n in labels]
    indices = {label: i for i, label in enumerate(labels)}
    form = [[Fraction() for _ in labels] for _ in labels]
    for j, label in enumerate(labels):
        for coefficient, word in words(record):
            for output, value in act(word, label).items():
                if output in indices:
                    i = indices[output]
                    form[i][j] += weights[i] * coefficient * value
    assert len(form) == 20 and form == list(map(list, zip(*form)))
    lower, upper = Fraction(1181, 50), Fraction(189, 8)
    counts = [negative_pivots(form, weights, c) for c in (lower, upper)]
    assert counts == [0, 1]
    # Every variational trial space considered contains the empty INNER seed.
    seed = indices[(0, 0, 0)]
    seed_rayleigh = form[seed][seed] / weights[seed]
    ceiling = Fraction(788, 25)
    assert seed_rayleigh < ceiling and ceiling - lower == Fraction(79, 10) < lower
    print(f"g=2 cutoff=3: negative pivots {counts} at {lower}, {upper}; "
          f"seed Rayleigh < {ceiling}; all seed-containing Ritz errors < {ceiling - lower}")


if __name__ == "__main__":
    main()
