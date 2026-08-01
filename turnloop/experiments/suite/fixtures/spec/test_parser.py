import pytest
from parser import parse_duration


def test_seconds():
    assert parse_duration("30s") == 30.0


def test_decimal_minutes():
    assert parse_duration("1.5m") == 90.0


def test_uppercase_and_whitespace():
    assert parse_duration(" 2H ") == 7200.0


def test_days():
    assert parse_duration("1d") == 86400.0


def test_bare_number_is_seconds():
    assert parse_duration("45") == 45.0


@pytest.mark.parametrize("bad", ["", "abc", "10y", "s"])
def test_invalid_raises(bad):
    with pytest.raises(ValueError):
        parse_duration(bad)
