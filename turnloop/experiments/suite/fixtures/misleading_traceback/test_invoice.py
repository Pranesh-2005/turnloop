from invoice import invoice_pretax_line


def test_invoice_pretax_includes_shipping():
    # The invoice line shows what's owed before tax: goods plus shipping.
    assert invoice_pretax_line(100, shipping=10) == 110
