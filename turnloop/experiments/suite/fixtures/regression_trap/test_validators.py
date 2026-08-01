from validators import is_valid_email, is_valid_username


def test_username_rejects_dot():
    assert not is_valid_username("bad.name")


def test_username_rejects_at_sign():
    assert not is_valid_username("bad@name")


def test_email_local_part_allows_dot():
    # "first.last@example.com" is a completely ordinary email address.
    assert is_valid_email("first.last@example.com")
