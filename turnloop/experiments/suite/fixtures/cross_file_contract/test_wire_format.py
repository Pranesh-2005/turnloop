from encode import encode_tags


def test_plain_tags_encode_unescaped():
    # No commas in the tags means no escaping is needed on the wire, and
    # downstream consumers (a log shipper) parse this exact shape.
    assert encode_tags(["prod", "web", "east"]) == "prod,web,east"
