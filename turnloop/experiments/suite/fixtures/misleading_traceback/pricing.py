"""Order composition, shared by checkout and invoicing.

`order_total_before_tax` is the combined amount a customer owes before tax:
subtotal plus any shipping fee. Invoicing needs that combined figure to show
what's due. Tax is a separate matter, though -- it is owed on the goods
only, since shipping is a pass-through fee in this jurisdiction.
"""


def order_total_before_tax(subtotal, shipping=0):
    return subtotal + shipping
