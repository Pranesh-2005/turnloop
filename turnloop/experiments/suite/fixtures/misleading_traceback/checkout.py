"""Checkout totals.

`test_checkout.py::test_total_with_shipping_is_not_taxed` fails with a total
that's too high. The traceback points at `total_with_tax` below, and the tax
rate and rounding there are both correct. The defect is that `checkout_total`
taxes the combined pretax figure from `pricing.py`, which includes shipping,
instead of taxing the goods portion alone.
"""

from pricing import order_total_before_tax

SALES_TAX_RATE = 8.5  # percentage points


def total_with_tax(amount):
    return round(amount * (1 + SALES_TAX_RATE / 100), 2)


def checkout_total(subtotal, shipping=0):
    pretax = order_total_before_tax(subtotal, shipping)
    return total_with_tax(pretax)
