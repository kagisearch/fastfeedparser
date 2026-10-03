"""Deterministic generator of RSS and Atom documents for parity tests.

The documents are well-formed but full of legal awkwardness: CDATA, character
references, comments inside text, mixed line endings, odd whitespace, repeated
and oddly-cased elements, media groups. `mutate` then breaks one in a single
place, to probe how malformed input is handled.
"""

import random

_NSDECL = (
    ' xmlns:atom="http://www.w3.org/2005/Atom"'
    ' xmlns:dc="http://purl.org/dc/elements/1.1/"'
    ' xmlns:content="http://purl.org/rss/1.0/modules/content/"'
    ' xmlns:media="http://search.yahoo.com/mrss/"'
    ' xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"'
    ' xmlns:x="urn:x" xmlns:a03="http://purl.org/atom/ns#"'
)
DATES = [
    "Mon, 02 Jan 2006 15:04:05 GMT",
    "Mon, 02 Jan 2006 15:04:05 +0000",
    "2 Jan 2006 15:04:05 -0700",
    "Tue, 31 Dec 2024 23:30:00 PST",
    "2024-01-15T10:30:00Z",
    "2024-01-15T10:30:00+02:00",
    "2024-01-15 10:30:00.123Z",
    "2024-01-15T10:30:00.123456-05:30",
    "2024-01-15",
    "2024-01-15T10:30:00",
    "January 5, 2024",
    "yesterday",
    "",
    " Mon, 02 Jan 2006 15:04:05 GMT ",
    "Mon,  02 Jan 2006 15:04:05 GMT",
    "2023-02-29T10:00:00Z",
    "Sat, 30 Dec 2023 24:00:00 GMT",
    "Mon, 02 Jan 2006 15:04:05 EST",
    "Mon, 02 Jan 2006 15:04:05 XYZ",
    "1700000000",
    "2024-06-01T00:00:00+00:00",
    "Wed, 45 Jan 2006 99:99:99 GMT",
    "2024-01-15T10:30:00.1Z",
    "0001-01-01T00:00:00+01:00",
    "9999-12-31T23:59:59-01:00",
]
_WORDS = [
    "lorem",
    "ipsum",
    "dolor",
    "caf\u00e9",
    "\u65e5\u672c",
    "a=b",
    "x  y",
    "tab\there",
    "\u00a0nbsp\u00a0",
    "<b>bold</b>",
    "AT&T",
    "5 > 3",
    '"q"',
    "it's",
    "\U0001f600",
]
_URLS = [
    None,
    "http://e.com/a",
    "https://e.com/b?x=1&y=2",
    "",
    " http://e.com/sp ",
    "/rel",
    "tag:x,2024:1",
    "http://e.com/\u00e9",
    "a\tb\nc",
]
_INTS = [None, "10", " 7 ", "", "1e3", "010", "1_000", "-3", "abc", "10.0"]
_SPLICES = [
    b"<",
    b">",
    b"&",
    b'"',
    b"'",
    b"=",
    b"/",
    b"]]>",
    b"<!--",
    b"-->",
    b"--",
    b"<?",
    b"?>",
    b"\r",
    b"\x00",
    b"\x0b",
    b"\xff",
    b"<![CDATA[",
    b"&nbsp;",
    b"&#0;",
    b"&#x1f;",
    b"<x>",
    b"</x>",
    b"<x",
    b" a=b",
    b' a="1" a="2"',
    b"<1>",
    b"<a:b:c/>",
    b"<!DOCTYPE x>",
    b"<!DOCTYPE x [<!ENTITY e 'v'>]>",
    b"<?xml version='1.0'?>",
    b"\xef\xbb\xbf",
    b"&e;",
    b" xmlns:q=''",
    b"<q:z/>",
]


def _esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class FeedGenerator:
    def __init__(self, seed):
        self.rnd = random.Random(seed)

    def _text(self):
        rnd = self.rnd
        r = rnd.random()
        count = rnd.choice([0, 1, 1, 3, 8, 60, 400])
        words = " ".join(rnd.choice(_WORDS) for _ in range(count))
        if r < 0.08:
            return ""
        if r < 0.14:
            return rnd.choice([" ", "\n  \n", "\t", "\u00a0"])
        if r < 0.4:
            tail = rnd.choice(["", " ]] ", "\r\n", " & "])
            return "<![CDATA[" + words.replace("]]>", "]] >") + tail + "]]>"
        if r < 0.5:
            refs = ["&#x41;", "&#65;", "&#10;", "&#13;", "&apos;&quot;", "\r\n", "\r"]
            return _esc(words) + rnd.choice(refs) + _esc(words[:20])
        if r < 0.56:
            nodes = ["<!-- c -->", "<?pi x?>", "<x:b>in</x:b>", "<b/>"]
            return _esc(words[:10]) + rnd.choice(nodes) + _esc(words[:10])
        if r < 0.62:
            return (
                "<![CDATA[" + words[:20] + "]]>" + _esc(words[:10]) + "<![CDATA[tail]]>"
            )
        if r < 0.7:
            lead = rnd.choice(["  ", "\n", "\u00a0", ""])
            return lead + _esc(words) + rnd.choice(["  ", "\n", "\u3000", ""])
        return _esc(words)

    def _attr(self, name, values):
        value = self.rnd.choice(values)
        if value is None:
            return ""
        value = value.replace("&", "&amp;").replace("<", "&lt;")
        quote = self.rnd.choice(['"', '"', "'"])
        value = value.replace(quote, "&quot;" if quote == '"' else "&apos;")
        return f" {name}={quote}{value}{quote}"

    def _el(self, tag, body=None, attrs=""):
        if body is None:
            body = self._text()
        if body == "" and self.rnd.random() < 0.5:
            return f"<{tag}{attrs}/>"
        return f"<{tag}{attrs}>{body}</{tag}>"

    def _thumbnail(self):
        attrs = self._attr("url", _URLS) + self._attr("width", _INTS)
        return self._el("media:thumbnail", "", attrs + self._attr("height", _INTS))

    def _media(self, depth=0):
        rnd = self.rnd
        inner = ""
        for _ in range(rnd.choice([0, 0, 1, 2, 3])):
            kind = rnd.randrange(6)
            if kind == 0:
                inner += self._el("media:title")
            elif kind == 1:
                inner += self._el("media:text")
            elif kind == 2:
                inner += self._el("media:description")
            elif kind == 3:
                scheme = self._attr("scheme", [None, "urn:ebu", ""])
                inner += self._el("media:credit", attrs=scheme)
            elif kind == 4:
                inner += self._thumbnail()
            elif depth < 2:
                inner += self._media(depth + 1)
        attrs = (
            self._attr("url", _URLS)
            + self._attr("type", [None, "image/png", ""])
            + self._attr("medium", [None, "image"])
            + self._attr("width", _INTS)
            + self._attr("height", _INTS)
        )
        return f"<media:content{attrs}>{inner}</media:content>"

    def _enclosure(self):
        attrs = self._attr("url", _URLS) + self._attr("type", [None, "audio/mpeg", ""])
        return self._el("enclosure", "", attrs + self._attr("length", _INTS))

    def _link(self, tag):
        rels = [
            None,
            "alternate",
            "self",
            "edit",
            "enclosure",
            "related",
            "replies",
            "",
        ]
        attrs = (
            self._attr("rel", rels)
            + self._attr("href", _URLS)
            + self._attr("link", [None, "http://e.com/l"])
            + self._attr("type", [None, "text/html"])
            + self._attr("title", [None, "T"])
        )
        return self._el(tag, "", attrs)

    def _content_attrs(self, types):
        attrs = self._attr("type", types) + self._attr("xml:lang", [None, "fr"])
        return attrs + self._attr("xml:base", [None, "http://b/"])

    def _rss_child(self):
        rnd, el = self.rnd, self._el
        kind = rnd.randrange(20)
        urls = [u for u in _URLS if u is not None]
        if kind == 0:
            titles = [
                "title",
                "title",
                "Title",
                "dc:title",
                "atom:title",
                "media:title",
            ]
            return el(rnd.choice(titles))
        if kind == 1:
            return el(rnd.choice(["link", "link", "Link"]), _esc(rnd.choice(urls)))
        if kind == 2:
            guids = urls + ["abc-123", "Mon, 02 Jan 2006 15:04:05 GMT"]
            permalink = self._attr("isPermaLink", [None, "true", "false", "TRUE"])
            return el("guid", _esc(rnd.choice(guids)), permalink)
        if kind == 3:
            names = ["description", "description", "summary", "media:description"]
            return el(rnd.choice(names))
        if kind == 4:
            return el("content:encoded", attrs=self._content_attrs([None, "html", ""]))
        if kind == 5:
            return el("content", attrs=self._attr("type", [None, "text", "html"]))
        if kind == 6:
            names = ["pubDate", "pubdate", "dc:date", "published", "issued", "updated"]
            names += ["lastBuildDate", "modified", "dc:modified"]
            return el(rnd.choice(names), _esc(rnd.choice(DATES)))
        if kind == 7:
            return el(rnd.choice(["author", "dc:creator", "creator", "Author"]))
        if kind == 8:
            names = ["atom:name", "name", "atom:email", "atom:uri"]
            inner = "".join(el(rnd.choice(names)) for _ in range(rnd.choice([0, 1, 2])))
            return f"<atom:author>{inner}</atom:author>"
        if kind == 9:
            return el("comments")
        if kind == 10:
            names = [
                "category",
                "Category",
                "dc:category",
                "x:category",
                "atom:category",
            ]
            attrs = self._attr("domain", [None, "d", ""]) + self._attr(
                "term", [None, "t"]
            )
            return el(rnd.choice(names), attrs=attrs)
        if kind == 11:
            return el("dc:subject")
        if kind == 12:
            return self._enclosure()
        if kind == 13:
            return self._link("atom:link")
        if kind == 14:
            return el("atom:id")
        if kind == 15:
            return self._media()
        if kind == 16:
            parts = [
                self._media(),
                el("media:description"),
                self._media(),
                el("media:credit"),
            ]
            return f"<media:group>{''.join(parts)}</media:group>"
        if kind == 17:
            return self._thumbnail()
        if kind == 18:
            return el(rnd.choice(["source", "x:unknown", "itunes", "a03:issued"]))
        return rnd.choice(["<!-- note -->", "<?pi data?>", "\n  "])

    def _rss_head_child(self):
        rnd, el = self.rnd, self._el
        kind = rnd.randrange(13)
        if kind == 0:
            return el("title")
        if kind == 1:
            return el("link", "http://e.com/")
        if kind == 2:
            return el("description")
        if kind == 3:
            return el("language", "en")
        if kind == 4:
            return el("lastBuildDate", _esc(rnd.choice(DATES)))
        if kind == 5:
            return el("category")
        if kind == 6:
            return el("generator")
        if kind == 7:
            parts = (
                el("url", "http://e.com/i.png")
                + el("title")
                + el("link", "http://e.com/")
            )
            return f"<image>{parts}</image>"
        if kind == 8:
            rel = self._attr("rel", [None, "self", "alternate", "hub"])
            return el("atom:link", "", rel + self._attr("href", _URLS))
        names = ["managingEditor", "webMaster", "dc:creator", "comments"]
        return el(names[kind - 9])

    def rss(self):
        rnd = self.rnd
        items = ""
        for _ in range(rnd.choice([0, 1, 1, 2, 3, 6])):
            attrs = (
                self._attr("xml:lang", [None, None, "en"])
                + self._attr("xml:base", [None, None, "http://b/"])
                + self._attr("rdf:about", [None, None, "http://e.com/about", ""])
            )
            children = "".join(self._rss_child() for _ in range(rnd.randint(0, 12)))
            items += f"<item{attrs}>{children}</item>"
            if rnd.random() < 0.2:
                items += rnd.choice(["\n", "<!-- between -->", self._el("x:between")])
        head = "".join(self._rss_head_child() for _ in range(rnd.randint(0, 8)))
        tail = "".join(
            self._el(rnd.choice(["title", "x:after", "link"]))
            for _ in range(rnd.choice([0, 0, 1]))
        )
        decl = rnd.choice(
            [
                "",
                '<?xml version="1.0" encoding="UTF-8"?>',
                "<?xml version='1.0'?>\n",
                '<?xml version="1.0" encoding="utf-8" standalone="yes"?>',
            ]
        )
        lang = self._attr("xml:lang", [None, "en-US"])
        channel = f"<channel{lang}>{head}{items}{tail}</channel>"
        return f'{decl}<rss version="2.0"{_NSDECL}>{channel}</rss>'

    def _atom_child(self):
        rnd, el = self.rnd, self._el
        kind = rnd.randrange(14)
        if kind == 0:
            return el("id")
        if kind == 1:
            return el("title", attrs=self._attr("type", [None, "html", "text"]))
        if kind == 2:
            return el("summary", attrs=self._attr("type", [None, "html"]))
        if kind == 3:
            return el("content", attrs=self._content_attrs([None, "html", "text", ""]))
        if kind == 4:
            names = ["published", "updated", "issued", "modified", "created"]
            return el(rnd.choice(names), _esc(rnd.choice(DATES)))
        if kind in (5, 6):
            return self._link("link")
        if kind == 7:
            names = ["name", "email", "uri", "x:name"]
            count = rnd.choice([0, 1, 2, 3])
            inner = "".join(el(rnd.choice(names)) for _ in range(count))
            return f"<author>{inner}</author>"
        if kind == 8:
            attrs = self._attr("term", [None, "t", ""]) + self._attr(
                "scheme", [None, "s"]
            )
            return el("category", "", attrs + self._attr("label", [None, "L"]))
        if kind == 9:
            return self._enclosure()
        if kind == 10:
            return self._media()
        if kind == 11:
            return self._thumbnail()
        if kind == 12:
            names = ["x:unknown", "source", "dc:creator", "atom:title", "a03:title"]
            return el(rnd.choice(names))
        return rnd.choice(["<!-- note -->", "<?pi data?>", "\n  "])

    def _atom_head_child(self):
        rnd, el = self.rnd, self._el
        kind = rnd.randrange(10)
        if kind == 0:
            return el("title")
        if kind == 1:
            return el("subtitle")
        if kind == 2:
            return el("id")
        if kind == 3:
            return el("updated", _esc(rnd.choice(DATES)))
        if kind == 4:
            rel = self._attr("rel", [None, "self", "alternate", "hub"])
            return el("link", "", rel + self._attr("href", _URLS))
        if kind == 5:
            return f"<author>{el('name')}</author>"
        if kind == 6:
            attrs = self._attr("version", [None, "1"]) + self._attr(
                "uri", [None, "http://g/"]
            )
            return el("generator", attrs=attrs)
        if kind == 7:
            return el("icon", "http://e.com/i.ico")
        if kind == 8:
            return el("logo", "http://e.com/l.png")
        return el("category", "", self._attr("term", [None, "t"]))

    def atom(self):
        rnd = self.rnd
        namespaces = [
            "http://www.w3.org/2005/Atom",
            "http://www.w3.org/2005/Atom",
            "https://www.w3.org/2005/Atom",
            "http://purl.org/atom/ns#",
        ]
        ns = rnd.choice(namespaces)
        entries = ""
        for _ in range(rnd.choice([0, 1, 1, 2, 3, 6])):
            attrs = self._attr("xml:lang", [None, None, "en"])
            attrs += self._attr("xml:base", [None, None, "http://b/"])
            children = "".join(self._atom_child() for _ in range(rnd.randint(0, 12)))
            entries += f"<entry{attrs}>{children}</entry>"
        head = "".join(self._atom_head_child() for _ in range(rnd.randint(0, 8)))
        if rnd.random() < 0.05:
            entries = f"<x:wrap>{entries}</x:wrap>"
        decl = rnd.choice(["", '<?xml version="1.0" encoding="UTF-8"?>'])
        return f'{decl}<feed xmlns="{ns}"{_NSDECL}>{head}{entries}</feed>'

    def document(self):
        """A well-formed RSS or Atom document as UTF-8 bytes."""
        text = self.rss() if self.rnd.random() < 0.6 else self.atom()
        return text.encode()

    def mutate(self, doc):
        """Break `doc` in one place: a splice, a deletion, or a truncation."""
        rnd = self.rnd
        r = rnd.random()
        pos = rnd.randrange(len(doc))
        if r < 0.45:
            return doc[:pos] + rnd.choice(_SPLICES) + doc[pos:]
        if r < 0.7:
            return doc[:pos] + doc[pos + rnd.choice([1, 1, 2, 5]) :]
        if r < 0.85:
            return doc[:pos]
        low, high = sorted((pos, rnd.randrange(len(doc))))
        return doc[:low] + doc[high:]
