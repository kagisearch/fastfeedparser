# fastfeedparser-core

Optional native extractor for [fastfeedparser](https://github.com/kagisearch/fastfeedparser).
When it is installed, `fastfeedparser.parse()` uses it for well-formed UTF-8
RSS and Atom feeds and keeps using lxml for everything else. Output is the
same either way.

You do not call this package directly.

## How it fits

`parse_entries(data, entry_cls, parse_date, ...)` makes one streaming pass
over the document with quick-xml and returns:

- the feed kind and Atom namespace,
- the document with its items cut out (fastfeedparser runs its existing
  feed-level code on that),
- one mapping per entry, already in the shape `parse()` returns.

If the document is anything other than well-formed UTF-8 RSS or Atom, it
returns a short reason string instead and fastfeedparser falls back to lxml.
That covers malformed XML, RDF, other encodings, xhtml content and internal
DTD subsets.

## Build and test

```bash
python -m venv .venv && . .venv/bin/activate
pip install maturin pytest
pip install -e ..                 # fastfeedparser itself
maturin develop --release         # build this crate into the venv
cargo test                        # Rust unit tests
pytest ..                         # includes tests/test_native_core.py
FASTFEEDPARSER_DISABLE_CORE=1 pytest ..   # the lxml path alone
cargo run --release --example bench -- ../benchmark_data
```

Set `FASTFEEDPARSER_DISABLE_CORE=1` to make fastfeedparser ignore the
extension at import time.

## Layout

| File | Role |
|---|---|
| `src/extract.rs` | Document-level state machine and the parse loop |
| `src/item.rs` | Elements inside one item or entry |
| `src/rss.rs`, `src/atom.rs` | Entry rules, ported from the Python functions |
| `src/media.rs` | Media RSS |
| `src/text.rs` | Text and attribute decoding, Python-compatible strip and int |
| `src/dates.rs` | ISO-8601 and RFC-822 fast paths |
| `src/validate.rs` | Strict well-formedness checks the tokenizer does not make |
| `src/lib.rs` | Python bindings |
