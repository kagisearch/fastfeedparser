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

**Date parsing cascade:** ISO-8601 → RFC-822 → dateutil → dateparser (slowest, LRU-cached). The two most common layouts skip the cascade: `Mon, 02 Jan 2006 15:04:05 GMT` is converted by character position, and a canonical UTC ISO timestamp is returned as it is.

**Native core:** `rust/` is the source of `fastfeedparser._core` (PyO3 + quick-xml), compiled into the package by `setup.py` through setuptools-rust as an optional extension: platform wheels contain it, the `py3-none-any` wheel and Rust-less source installs do not. When it is importable, `_parse_with_core()` uses it for well-formed UTF-8 RSS and Atom; it returns a reason string for anything else and the lxml path runs as before. The two paths must produce identical output: `tests/test_native_core.py` compares them on the fixtures and on generated feeds (`tests/feedgen.py`). Any change to entry extraction in `main.py` needs the matching change in `rust/src/rss.rs`, `atom.rs` or `media.rs`.

**Performance patterns:**
- lxml recover parser in one pass; a strict parse only decides whether a malformed-looking document needs body repair
- One pair of lxml parsers per thread (`_THREAD_XML_PARSERS`); lxml locks a parser for a whole parse, so a shared one serializes threads
- Threads keep their own parsers only while the trees in flight are estimated to fit `_MAX_TREE_BYTES_IN_FLIGHT` (32 MB; `_estimated_tree_bytes` is about 4x the document plus 200 bytes per tag). A document over that is parsed with the one shared pair (`_SHARED_XML_PARSERS`), whose lxml lock serializes the parse as the single pair in 0.6.3 did. The lock does not cover the tree's lifetime, so a feed whose entries are slow to read can still overlap with the next one, as on 0.6.3. Nothing waits or locks in Python (`_xml_parsers`)
- Large CDATA sections are lifted out of the bytes lxml parses and decoded in Python (`_lift_large_cdata`); every reader of description/content/summary text restores them. If libxml2 reports any error or reads another encoding, the document is parsed again whole. Skipped while another thread is parsing, because the scan holds the GIL
- A document over 256 KB whose first element is not a feed (HTML page, sitemap, OPML) is only parsed at its head (`_raise_for_non_feed_head`), and a meta-refresh or error message is only looked for in the first 256 KB of a page (`_HTML_HEAD_BYTES`)
- Pre-compiled regex (`_RE_*` constants)
- LRU-cached slow parsers (`_slow_dateutil_parse`, `_slow_dateparser`)

## Testing

Snapshot testing: feed files in `tests/integration/` compared against `.json` expected output.

To add a test case:
1. Add feed file to `tests/integration/`
2. Run `pytest` - generates expected `.json` on first run
3. Verify output, commit both files
