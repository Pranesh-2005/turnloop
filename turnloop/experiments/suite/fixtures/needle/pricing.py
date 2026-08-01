"""Price bands. Contains the off-by-one error."""

BANDS = [(0, 10, 1.0), (10, 50, 0.9), (50, 200, 0.8), (200, 10_000, 0.7)]


def band_for(quantity):
    # Off-by-one: the final band is never reachable because the loop stops one
    # short of the end of BANDS.
    for i in range(len(BANDS) - 1):
        low, high, multiplier = BANDS[i]
        if low <= quantity < high:
            return multiplier
    return 1.0


def price(unit_cost, quantity):
    return unit_cost * quantity * band_for(quantity)
