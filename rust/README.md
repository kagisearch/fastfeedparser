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

## Releasing

Release files come from CI (`.github/workflows/wheels.yml`), not from a local
build. A wheel built locally covers one platform and is tagged for the macOS
version the local Python was built for (`macosx_26_0_arm64` on a current Mac),
so almost nobody could install it.

1. Bump the version in `setup.cfg` and `src/fastfeedparser/__init__.py`,
   commit, and tag the commit:

   ```bash
   git tag -a vX.Y.Z -m "vX.Y.Z - summary"
   git push origin main vX.Y.Z
   ```

2. The tag push starts the `wheels` workflow. It builds and tests one wheel
   per platform, the pure wheel and the sdist, then gathers them in a single
   `dist` artifact. Wait for it to finish:

   ```bash
   gh run list --workflow wheels.yml -L 3
   gh run watch <run-id> --exit-status
   ```

3. Download the artifact, check it, upload it:

   ```bash
   gh run download <run-id> -n dist -D release-dist
   twine check release-dist/*
   twine upload release-dist/*
   ```

   A full set is 9 files: the sdist, the pure wheel, four Linux wheels
   (manylinux and musllinux, x86_64 and aarch64), two macOS wheels and one
   Windows wheel.

If one platform job fails, the `dist` job is skipped, but every job that
passed still has its own artifact (`wheels-<os>`, `wheels-pure-and-sdist`);
`gh run download <run-id>` without `-n` fetches them all. PyPI never lets a
file be replaced, but it accepts more files for a version that already exists,
so a missing wheel can be uploaded later. To rebuild without a new tag, run
the workflow by hand: `gh workflow run wheels.yml --ref <branch>`.

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
