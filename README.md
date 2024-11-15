# FastFeedParser

A high-performance RSS, Atom, and RDF feed parser for Python. FastFeedParser is designed to be fast, memory-efficient, and easy to use while providing comprehensive feed parsing capabilities.

### Why FastFeedParser?

The main advantage of FastFeedParser over the traditional feedparser library is its lightweight design and exceptional performance - benchmarks show it's 10x-100x faster than feedparser while maintaining a familiar API. This dramatic speed improvement is achieved through:

- Efficient XML parsing using lxml
- Optimized memory usage
- Minimal dependencies
- Streamlined codebase focused on core functionality

## Features

- Fast parsing of RSS 2.0, Atom 1.0, and RDF/RSS 1.0 feeds
- Robust error handling and encoding detection
- Support for media content and enclosures
- Automatic date parsing with timezone handling
- Clean, Pythonic API similar to feedparser
- Comprehensive handling of feed metadata
- Support for various feed extensions (Media RSS, Dublin Core, etc.)

## Installation

```bash
pip install fastfeedparser
```

## Quick Start

```python
import fastfeedparser

# Parse from URL
feed = fastfeedparser.parse_url('https://example.com/feed.xml')

# Parse from string
xml_content = '''<?xml version="1.0"?>
<rss version="2.0">
    <channel>
        <title>Example Feed</title>
        ...
    </channel>
</rss>'''
feed = fastfeedparser.parse(xml_content)

# Access feed information
print(feed.feed.title)
print(feed.feed.link)

# Access entries
for entry in feed.entries:
    print(entry.title)
    print(entry.link)
    print(entry.published)
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

- `parse(xml_content)`: Parse feed from a string or bytes
- `parse_url(url)`: Parse feed from a URL
- `fetch_url(url)`: Fetch content from a URL

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
- httpx
- lxml
- parsedatetime
- python-dateutil

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

Inspired by the Universal Feed Parser (feedparser) project, FastFeedParser aims to provide a modern, high-performance alternative while maintaining a familiar API.
