"""The Order model."""


class Order:
    def __init__(self, order_id, total):
        self.order_id = order_id
        self.total = total
        self.cancelled = False

    def __repr__(self):
        return f"Order({self.order_id!r}, total={self.total})"
