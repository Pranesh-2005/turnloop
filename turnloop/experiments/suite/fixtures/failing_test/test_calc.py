from calc import add, average, percent_change


def test_add():
    assert add(2, 3) == 5


def test_average():
    assert average([1, 2, 3]) == 2


def test_average_of_nothing_is_zero():
    assert average([]) == 0.0


def test_percent_change():
    assert percent_change(100, 150) == 50
