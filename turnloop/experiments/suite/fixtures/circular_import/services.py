"""Order-processing services, built on top of the model in models.py."""

from models import Order

_ORDERS = {}


def create_order(order_id, total):
    order = Order(order_id, total)
    _ORDERS[order_id] = order
    return order


def get_order(order_id):
    return _ORDERS.get(order_id)
