from billing import compute_total, invoice

ITEMS = [{"price": 2.0, "qty": 3}, {"price": 1.5, "qty": 2}]


def test_compute_total():
    assert compute_total(ITEMS) == 9.0


def test_invoice():
    assert invoice(ITEMS, 0.0) == 9.0
