from checkout import checkout_total


def test_total_without_shipping():
    # $100 of goods at 8.5% tax, no shipping.
    assert checkout_total(100) == 108.50


def test_total_with_shipping_is_not_taxed():
    # $100 of goods plus a $10 pass-through shipping fee: tax applies to
    # the $100 of goods only.
    assert checkout_total(100, shipping=10) == 118.50
