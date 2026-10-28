#!/usr/bin/env python3
"""Estimate the PSI noise floor: PSI between two samples of the SAME distribution.

Shows why analysis.py uses 500 records per window. With few records, PSI over
10 bins is inflated by sampling noise alone and can cross the 0.25 threshold
without any real drift. Standard library only.

Usage: python noise_floor.py
"""
import random
import statistics

from analysis import BINS, PSI_MODERATE, PSI_SIGNIFICANT, population_stability_index

TRIALS = 400


def main():
    print(f"PSI between identical distributions ({TRIALS} trials, {BINS} bins)")
    print(f"{'records':>8}{'mean':>8}{'p95':>8}{'max':>8}{'>=0.10':>9}{'>=0.25':>9}")
    for n in (100, 200, 500, 1000):
        values = []
        for trial in range(TRIALS):
            rng = random.Random(trial)
            a = [rng.gauss(320, 60) for _ in range(n)]
            b = [rng.gauss(320, 60) for _ in range(n)]
            values.append(population_stability_index(a, b, BINS))
        values.sort()
        p95 = values[int(0.95 * TRIALS)]
        over_mod = sum(v >= PSI_MODERATE for v in values) / TRIALS
        over_sig = sum(v >= PSI_SIGNIFICANT for v in values) / TRIALS
        print(f"{n:>8}{statistics.fmean(values):>8.3f}{p95:>8.3f}{values[-1]:>8.3f}"
              f"{over_mod:>9.0%}{over_sig:>9.0%}")


if __name__ == "__main__":
    main()
