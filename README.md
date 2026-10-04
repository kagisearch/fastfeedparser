# FastFeedParser

A high-performance feed parser for Python that handles RSS, Atom, and RDF. Built for speed, efficiency, and ease of use while delivering complete parsing capabilities.

### Why FastFeedParser?

It's about 35x faster (check included `benchmark.py`) than popular feedparser
library while keeping a familiar API. This speed comes from:

- lxml for efficient XML parsing
- Smart memory management  
- Minimal dependencies
- Focused, streamlined code

Powers feed processing for [Kagi Small Web](https://github.com/kagisearch/smallweb), handling processing of thousands of feeds at scale.


## Features

- Fast parsing of RSS 2.0, Atom 1.0, and RDF/RSS 1.0 feeds
- Robust error handling and encoding detection
- Support for media content and enclosures
- Automatic date parsing and standardization to UTC ISO 8601 format
- Clean, Pythonic API similar to feedparser
- Comprehensive handling of feed metadata
- Support for various feed extensions (Media RSS, Dublin Core, etc.)


## Installation

```bash
pip install fastfeedparser
```

### Native core

The package includes a Rust extension (`fastfeedparser._core`, source in
`rust/`) that extracts entries from well-formed UTF-8 RSS and Atom feeds.
`parse()` uses it automatically and falls back to lxml for everything else;
the output is the same either way.

You do not need Rust to install. Wheels for CPython 3.9+ on Linux, macOS and
Windows carry the compiled extension. Anywhere else pip installs the
pure-Python wheel, which parses with lxml. Installing from source builds the
extension if a Rust toolchain is present and skips it if not.

Set `FASTFEEDPARSER_DISABLE_CORE=1` to ignore the extension. See
`rust/README.md`.

## Quick Start

```python
import fastfeedparser

# Parse from URL
myfeed = fastfeedparser.parse('https://example.com/feed.xml')

# Parse from string
xml_content = '''<?xml version="1.0"?>
<rss version="2.0">
    <channel>
        <title>Example Feed</title>
        ...
    </channel>
</rss>'''
myfeed = fastfeedparser.parse(xml_content)

# Access feed global information
print(myfeed.feed.title)
print(myfeed.feed.link)

# Access feed entries
for entry in myfeed.entries:
    print(entry.title)
    print(entry.link)
    print(entry.published)
```

## Run Benchmark

```bash
python benchmark.py
```

This will run benchmark on a number of feeds with output looking like this

```
Testing https://gessfred.xyz/rss.xml
FastFeedParser: 17 entries in 0.004s
Feedparser: 17 entries in 0.098s
Speedup: 26.3x

Testing https://fanf.dreamwidth.org/data/rss
FastFeedParser: 25 entries in 0.005s
Feedparser: 25 entries in 0.087s
Speedup: 17.9x

Testing https://jacobwsmith.xyz/feed.xml
FastFeedParser: 121 entries in 0.030s
Feedparser: 121 entries in 0.166s
Speedup: 5.5x

Testing https://bernsteinbear.com/feed.xml
FastFeedParser: 11 entries in 0.007s
Feedparser: 11 entries in 0.339s
Speedup: 50.1x
```

And publish a full report looking like this
```
Summary:
--------------------------------------------------
Total wall-clock time: 40.90s
Successfully tested 200/200 feeds

FastFeedParser:
  Total entries: 6600
  Total parsing time: 0.35s
  Average per feed: 0.002s
  Feeds/sec: 571.3

Feedparser:
  Total entries: 6555
  Total parsing time: 13.19s
  Average per feed: 0.066s
  Feeds/sec: 15.2

Speedup: FastFeedParser is 37.7x faster

OUTLIERS: Entry Count Mismatches (2 feeds)
--------------------------------------------------
  https://dylanharris.org/feed-me.rss
    FastFeedParser: 35 entries
    Feedparser: 0 entries
    Difference: +35
  https://humanwhocodes.com/feeds/all.json
    FastFeedParser: 10 entries
    Feedparser: 0 entries
    Difference: +10
```

## Key Features

### Feed Types Support
- RSS 2.0
- Atom 1.0
- RDF/RSS 1.0

### Content Handling
- Automatic encoding detection
- HTML content parsing
- Media content extraction
- Enclosure handling

### Metadata Support
- Feed title, link, and description
- Publication dates
- Author information
- Categories and tags
- Media content and thumbnails

## API Reference

### Main Functions

- `parse(source, *, include_content=True, include_tags=True, include_media=True, include_enclosures=True)`: Parse feed from a URL/XML/JSON source, with optional field extraction toggles for faster parsing.


### Feed Object Structure

The parser returns a `FastFeedParserDict` object with two main sections:

- `feed`: Contains feed-level metadata
- `entries`: List of feed entries

Each entry contains:
- `title`: Entry title
- `link`: Entry URL
- `description`: Entry description/summary
- `published`: Publication date
- `author`: Author information
- `content`: Full content
- `media_content`: Media attachments
- `enclosures`: Attached files

## Requirements

- Python 3.7+
- lxml
- python-dateutil

Optional extras:

- `brotli` (`pip install fastfeedparser[brotli]`) for `Content-Encoding: br`
- `dateparser` (`pip install fastfeedparser[dateparser]`) for the slowest date parsing fallback
- `pip install fastfeedparser[full]` for both

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Inspired by the [feedparser](https://github.com/kurtmckee/feedparser) project, FastFeedParser aims to provide a modern, high-performance alternative while maintaining a familiar API.
