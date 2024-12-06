import json
from pathlib import Path

import pytest

from fastfeedparser import parse

_TESTS_DIR = Path(__file__).parent
_INTEGRATION_DIR = _TESTS_DIR.joinpath("integration")


def pytest_generate_tests(metafunc: pytest.Metafunc):
    metafunc.parametrize("feed_path", sorted(_INTEGRATION_DIR.glob("*.xml")))


def test_integration(feed_path: Path):
    feed = feed_path.read_bytes()
    feed_parsed = parse(feed)
    expected_path = feed_path.with_suffix(".json")
    try:
        expected = json.loads(expected_path.read_text())
    except FileNotFoundError:
        expected_path.write_text(
            json.dumps(feed_parsed, ensure_ascii=False, indent=2, sort_keys=True)
        )
        return
    assert feed_parsed == expected
