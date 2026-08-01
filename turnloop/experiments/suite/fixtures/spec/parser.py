def parse_duration(text):
    """Parse a duration string into seconds.

    Accepts a number followed by a unit suffix: `s` for seconds, `m` for minutes,
    `h` for hours, `d` for days. The number may be an integer or a decimal.
    Whitespace around the value is ignored, and the suffix is case-insensitive.

    A bare number with no suffix is interpreted as seconds.

    Examples:
        parse_duration("30s")   -> 30.0
        parse_duration("1.5m")  -> 90.0
        parse_duration(" 2H ")  -> 7200.0
        parse_duration("45")    -> 45.0

    Raises ValueError for an empty string, a non-numeric value, or an unknown
    suffix.
    """
    raise NotImplementedError
