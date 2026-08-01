from decode import decode_tags
from encode import encode_tags


def test_round_trip_simple_tags():
    assert decode_tags(encode_tags(["prod", "web"])) == ["prod", "web"]


def test_round_trip_tag_with_comma():
    tags = ["client,vip", "renewal"]
    assert decode_tags(encode_tags(tags)) == tags
