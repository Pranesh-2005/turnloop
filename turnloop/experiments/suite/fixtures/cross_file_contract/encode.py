"""Encodes a list of tags into the wire format shared with decode.py.

Tags are joined with `,`. A literal comma inside a tag must be escaped so
decode.py can tell a separator from data — this module and decode.py have to
change together if the escaping rule ever changes.
"""


def encode_tags(tags):
    return ",".join(tags)
