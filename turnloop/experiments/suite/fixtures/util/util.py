"""Small helpers."""


def chunk(items, size):
    return [items[i : i + size] for i in range(0, len(items), size)]


def flatten(nested):
    return [item for group in nested for item in group]
