"""Shared form-field validators.

Usernames and the local part of an email address both get checked against a
"token" shape, so they share one pattern.
"""

import re

# Both usernames and email local-parts are checked against this pattern.
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_]+$")


def is_valid_username(name):
    """A username is 3-20 characters of letters, digits, or underscore only."""
    return bool(_TOKEN_RE.match(name)) and 3 <= len(name) <= 20


def is_valid_email(address):
    """An email address is local@domain, where local reuses the username token rule."""
    if "@" not in address:
        return False
    local, _, domain = address.partition("@")
    return bool(_TOKEN_RE.match(local)) and "." in domain
