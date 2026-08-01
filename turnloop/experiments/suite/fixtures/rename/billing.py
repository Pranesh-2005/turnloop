def compute_total(items):
    return sum(item["price"] * item["qty"] for item in items)


def invoice(items, tax_rate=0.2):
    subtotal = compute_total(items)
    return subtotal * (1 + tax_rate)
