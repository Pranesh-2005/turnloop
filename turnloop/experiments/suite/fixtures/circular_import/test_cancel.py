from services import cancel_order, create_order, get_order


def test_cancel_marks_order():
    create_order("o1", 42)
    cancel_order("o1")
    assert get_order("o1").cancelled is True


def test_create_leaves_order_active():
    create_order("o2", 10)
    assert get_order("o2").cancelled is False
