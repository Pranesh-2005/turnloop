"""A small calculator with one genuine bug."""


def add(a, b):
    return a + b


def average(values):
    # The bug: an empty list should return 0.0, not raise.
    return sum(values) / len(values)


def percent_change(old, new):
    return (new - old) / old * 100
