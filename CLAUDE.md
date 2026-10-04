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

# Native core: `pip install -e .` also compiles rust/ into
# src/fastfeedparser/_core.abi3.so when a Rust toolchain is present.
# Re-run it after changing Rust code.
(cd rust && cargo test)                      # Rust unit tests
FASTFEEDPARSER_DISABLE_CORE=1 pytest         # test the lxml path alone
FASTFEEDPARSER_PURE=1 python -m build        # build without the extension
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

**Native core:** `rust/` is the source of `fastfeedparser._core` (PyO3 + quick-xml), compiled into the package by `setup.py` through setuptools-rust as an optional extension: platform wheels contain it, the `py3-none-any` wheel and Rust-less source installs do not. When it is importable, `_parse_with_core()` uses it for well-formed UTF-8 RSS and Atom; it returns a reason string for anything else and the lxml path runs as before. The two paths must produce identical output: `tests/test_native_core.py` compares them on the fixtures and on generated feeds (`tests/feedgen.py`). Any change to entry extraction in `main.py` needs the matching change in `rust/src/rss.rs`, `atom.rs` or `media.rs`.

**Performance patterns:**
- lxml recover parser in one pass; a strict parse only decides whether a malformed-looking document needs body repair
- Pre-compiled regex (`_RE_*` constants)
- LRU-cached slow parsers (`_slow_dateutil_parse`, `_slow_dateparser`)

## Testing

Snapshot testing: feed files in `tests/integration/` compared against `.json` expected output.

To add a test case:
1. Add feed file to `tests/integration/`
2. Run `pytest` - generates expected `.json` on first run
3. Verify output, commit both files
