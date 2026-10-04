# The native core

Rust source of `fastfeedparser._core`, the extension module inside the
`fastfeedparser` package. It is not a separate distribution: the top-level
`setup.py` compiles it into the package with setuptools-rust.

When the module is present, `fastfeedparser.parse()` uses it for well-formed
UTF-8 RSS and Atom feeds and keeps using lxml for everything else. Output is
the same either way. You do not call it directly.

## How it fits

`parse_entries(data, entry_cls, parse_date, unescape, ...)` makes one
streaming pass over the document with quick-xml and returns:

- the feed kind and Atom namespace,
- the document with its items cut out (fastfeedparser runs its existing
  feed-level code on that),
- one mapping per entry, already in the shape `parse()` returns.

If the document is anything other than well-formed UTF-8 RSS or Atom, it
returns a short reason string instead and fastfeedparser falls back to lxml.
That covers malformed XML, RDF, other encodings, xhtml content and internal
DTD subsets.

## How it is shipped

One package, three kinds of file per release:

| File | Contains the extension | Who gets it |
|---|---|---|
| Platform wheels (`cp39-abi3-...`) | yes | CPython 3.9+ on the platforms we build for |
| Pure wheel (`py3-none-any`) | no | every other interpreter and platform |
| sdist | sources | source installs: built if a Rust toolchain is present, skipped if not |

pip picks the most specific wheel that fits, so nobody needs Rust to install
and no install fails for lack of it. `FASTFEEDPARSER_PURE=1` makes a build
skip the extension, which is how the pure wheel is produced.

## Build and test

```bash
pip install -e ..                 # builds the extension if Rust is available
cargo test                        # Rust unit tests
pytest ..                         # includes tests/test_native_core.py
FASTFEEDPARSER_DISABLE_CORE=1 pytest ..   # the lxml path alone
cargo run --release --example bench -- ../benchmark_data
```

After changing Rust code, run `pip install -e ..` again to rebuild. Set
`FASTFEEDPARSER_DISABLE_CORE=1` to make fastfeedparser ignore the extension
at import time.

## Layout

| File | Role |
|---|---|
| `src/extract.rs` | Document-level state machine and the parse loop |
| `src/item.rs` | Elements inside one item or entry |
| `src/rss.rs`, `src/atom.rs` | Entry rules, ported from the Python functions |
| `src/media.rs` | Media RSS |
| `src/synth.rs` | Description synthesized from content |
| `src/text.rs` | Text and attribute decoding, Python-compatible strip and int |
| `src/dates.rs` | ISO-8601 and RFC-822 fast paths |
| `src/validate.rs` | Strict well-formedness checks the tokenizer does not make |
| `src/lib.rs` | Python bindings |
