"""The cached element getter must resolve paths the way Element.find() does."""

import pytest
from lxml import etree

from fastfeedparser.main import _cached_element_value_factory

_DOC = (
    '<c xmlns:a="urn:a">'
    "<a:t>namespaced</a:t><t>plain</t>"
    "<a:p><a:t>nested namespaced</a:t></a:p><p><t>nested plain</t></p>"
    "</c>"
)


@pytest.mark.parametrize(
    "path",
    [
        "t",
        "{urn:a}t",
        "{*}t",
        "{}t",
        "p/t",
        "{urn:a}p/{urn:a}t",
        "{*}p/{*}t",
        "{}p/{}t",
        "{urn:b}t",
        "missing",
    ],
)
def test_getter_matches_find(path):
    root = etree.fromstring(_DOC)
    found = root.find(path)
    expected = None if found is None else found.text
    assert _cached_element_value_factory(root)(path) == expected
