# FastFeedParser

A high-performance feed parser for Python that handles RSS, Atom, and RDF. Built for speed, efficiency, and ease of use while delivering complete parsing capabilities.

### Why FastFeedParser?

It's 10x-100x faster than traditional feedparser while keeping a familiar API. This speed comes from:

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

- `parse(source)`: Parse feed from a source that can be URL or a string


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
