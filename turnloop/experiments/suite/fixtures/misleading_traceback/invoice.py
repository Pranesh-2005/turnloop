"""Customer-facing invoice line, built on the same pricing helper as checkout."""

from pricing import order_total_before_tax


def invoice_pretax_line(subtotal, shipping=0):
    """What the invoice shows as owed before tax: goods plus shipping."""
    return order_total_before_tax(subtotal, shipping)
