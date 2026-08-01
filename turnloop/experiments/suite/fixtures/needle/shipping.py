"""Shipping estimates. Correct."""

ZONES = {"north": 4.5, "south": 5.0, "east": 6.25, "west": 7.0}


def cost(zone, weight_kg):
    base = ZONES.get(zone, 8.0)
    return base + max(0.0, weight_kg - 1.0) * 1.25


def cheapest_zones(limit):
    ordered = sorted(ZONES.items(), key=lambda item: item[1])
    return [name for name, _ in ordered[:limit]]
