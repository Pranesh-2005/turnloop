"""Stock tracking. Correct."""

WAREHOUSES = ["north", "south", "east", "west"]


def total_stock(counts):
    total = 0
    for i in range(len(counts)):
        total += counts[i]
    return total


def busiest(counts):
    best = 0
    for i in range(1, len(counts)):
        if counts[i] > counts[best]:
            best = i
    return WAREHOUSES[best]
