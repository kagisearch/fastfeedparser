# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastFeedParser is a high-performance Python library for parsing RSS, Atom, RDF, and JSON feeds. ~10x faster than feedparser while maintaining a similar API. Used in production by [Kagi Small Web](https://github.com/kagisearch/smallweb).

## Commands

```bash
pip install -e .              # Install dependencies
pytest                        # Run all tests
pytest -k "test_name"         # Run tests matching pattern
python benchmark.py           # Benchmark against feedparser
python benchmark.py -s        # Benchmark fastfeedparser only
```

## Architecture

Single-file parser: `src/fastfeedparser/main.py`

**Entry point:** `parse(source)` - accepts URL or XML/JSON string/bytes

**Feed type detection order:** RSS 2.0 → Atom 1.0 → RDF/RSS 1.0 → JSON Feed

**Internal parsing functions:**
- `_parse_rss()` - RSS 2.0 with fallback to Atom-style entries
- `_parse_atom()` - Atom 1.0
- `_parse_rdf()` - RDF/RSS 1.0
- `_parse_json_feed()` - JSON Feed 1.0/1.1

**Date parsing cascade:** ISO-8601 → RFC-822 → dateutil → dateparser (slowest, LRU-cached)

**Performance patterns:**
- lxml with strict parser first, recover parser as fallback
- Pre-compiled regex (`_RE_*` constants)
- LRU-cached slow parsers (`_slow_dateutil_parse`, `_slow_dateparser`)

## Testing

Snapshot testing: feed files in `tests/integration/` compared against `.json` expected output.

To add a test case:
1. Add feed file to `tests/integration/`
2. Run `pytest` - generates expected `.json` on first run
3. Verify output, commit both files
