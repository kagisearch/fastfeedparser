import json
from pathlib import Path

import pytest

from fastfeedparser import parse

_TESTS_DIR = Path(__file__).parent
_INTEGRATION_DIR = _TESTS_DIR.joinpath("integration")


def pytest_generate_tests(metafunc: pytest.Metafunc):
    # Include both XML and JSON feed files
    xml_files = list(_INTEGRATION_DIR.glob("*.xml"))

    # Only include JSON files that don't have a corresponding .xml file
    # (to avoid treating expected output files as feed inputs)
    json_files = []
    for f in _INTEGRATION_DIR.glob("*.json"):
        # Skip .expected.json files
        if f.name.endswith(".expected.json"):
            continue
        # Skip .json files that have a corresponding .xml file (these are expected outputs)
        xml_equivalent = f.with_suffix(".xml")
        if xml_equivalent.exists():
            continue
        # This is an actual JSON feed file
        json_files.append(f)

    all_feeds = sorted(xml_files + json_files)
    metafunc.parametrize("feed_path", all_feeds)


def test_integration(feed_path: Path):
    feed = feed_path.read_bytes()
    feed_parsed = parse(feed)

    # For JSON feeds, use .expected.json extension for expected output
    # For XML feeds, use .json extension for expected output
    if feed_path.suffix == ".json":
        expected_path = feed_path.with_suffix(".expected.json")
    else:
        expected_path = feed_path.with_suffix(".json")

    try:
        expected = json.loads(expected_path.read_text())
    except FileNotFoundError:
        expected_path.write_text(
            json.dumps(feed_parsed, ensure_ascii=False, indent=2, sort_keys=True)
        )
        return
    assert feed_parsed == expected
