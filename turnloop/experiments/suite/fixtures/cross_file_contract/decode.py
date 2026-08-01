"""Decodes the wire format written by encode.py. See encode.py for the escaping rule."""


def decode_tags(line):
    if not line:
        return []
    return line.split(",")
