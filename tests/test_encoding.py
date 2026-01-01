from fastfeedparser import parse


def test_parse_str_with_non_utf8_xml_declaration():
    xml = (
        '<?xml version="1.0" encoding="iso-8859-1"?>'
        '<rss version="2.0">'
        "<channel>"
        "<title>café</title>"
        "<item><title>café</title></item>"
        "</channel>"
        "</rss>"
    )
    feed = parse(xml)
    assert feed.feed.title == "café"
    assert feed.entries[0].title == "café"


def test_parse_bytes_with_non_utf8_encoding():
    xml_bytes = (
        b'<?xml version="1.0" encoding="iso-8859-1"?>'
        b'<rss version="2.0">'
        b"<channel>"
        b"<title>caf\xe9</title>"
        b"<item><title>caf\xe9</title></item>"
        b"</channel>"
        b"</rss>"
    )
    feed = parse(xml_bytes)
    assert feed.feed.title == "café"
    assert feed.entries[0].title == "café"

