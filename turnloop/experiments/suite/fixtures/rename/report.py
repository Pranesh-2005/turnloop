from billing import compute_total


def summary(items):
    return f"{len(items)} items, total {compute_total(items):.2f}"
