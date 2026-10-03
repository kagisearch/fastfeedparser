//! Streaming extraction: one pass over the document, one `Entry` per item.
use quick_xml::events::{BytesStart, Event};
use quick_xml::name::ResolveResult;
use quick_xml::reader::NsReader;

use crate::atom::AtomAcc;
use crate::item::{Frame, ItemAcc, Role};
use crate::media::MediaAcc;
use crate::model::{
    DateFn, Document, Entry, FeedKind, ItemAttrs, Options, Stop, UnescapeFn, Unhandled,
};
use crate::ns::{Ns, RDF};
use crate::rss::RssAcc;
use crate::text::{attr_value, check_text, push_cdata, push_text};
use crate::validate::{self, attribute_name, read_attrs};

pub(crate) struct State<'o> {
    pub(crate) opts: &'o Options,
    pub(crate) depth: usize,
    pub(crate) kind: Option<FeedKind>,
    pub(crate) feed_ns: Ns,
    pub(crate) root_closed: bool,
    pub(crate) doctype_seen: bool,
    pub(crate) channel_seen: bool,
    pub(crate) in_channel: bool,
    /// Depth of the open item element, 0 when outside any item.
    pub(crate) item_depth: usize,
    pub(crate) item_start: usize,
    pub(crate) item_attrs: ItemAttrs,
    pub(crate) frames: Vec<Frame>,
    pub(crate) acc: Option<ItemAcc>,
    pub(crate) media: MediaAcc,
    pub(crate) ranges: Vec<(usize, usize)>,
    pub(crate) entries: Vec<Entry>,
}

pub(crate) type Reader<'i> = NsReader<&'i [u8]>;

impl State<'_> {
    fn start(
        &mut self,
        ns: Ns,
        e: &BytesStart,
        pos: usize,
        reader: &Reader,
    ) -> Result<(), Unhandled> {
        if ns == Ns::UnknownPrefix {
            return Err(Unhandled("undeclared element prefix"));
        }
        validate::name(e.name().as_ref())?;
        validate::attributes_syntax(e.attributes_raw())?;
        self.depth += 1;
        let name = e.local_name();
        let local = name.as_ref();
        if self.item_depth != 0 {
            return self.start_in_item(ns, local, e, reader);
        }
        if self.depth == 1 {
            read_attrs(reader, e, [])?;
            return self.start_root(ns, local);
        }
        let begins_item = match self.kind {
            Some(FeedKind::Rss) => {
                self.depth == 3 && self.in_channel && ns == Ns::None && local == b"item"
            }
            Some(FeedKind::Atom) => ns == self.feed_ns && local == b"entry",
            None => false,
        };
        if begins_item {
            return self.begin_item(e, pos, reader);
        }
        read_attrs(reader, e, [])?;
        let is_channel = self.kind == Some(FeedKind::Rss)
            && self.depth == 2
            && ns == Ns::None
            && local == b"channel";
        if is_channel && !self.channel_seen {
            self.channel_seen = true;
            self.in_channel = true;
        }
        Ok(())
    }

    fn start_root(&mut self, ns: Ns, local: &[u8]) -> Result<(), Unhandled> {
        if self.root_closed {
            return Err(Unhandled("second root element"));
        }
        if ns == Ns::None && local == b"rss" {
            self.kind = Some(FeedKind::Rss);
        } else if ns.atom_uri().is_some() && local == b"feed" {
            self.kind = Some(FeedKind::Atom);
            self.feed_ns = ns;
        } else {
            return Err(Unhandled("root element"));
        }
        Ok(())
    }

    fn begin_item(&mut self, e: &BytesStart, pos: usize, reader: &Reader) -> Result<(), Unhandled> {
        let mut attrs = ItemAttrs::default();
        for attr in e.attributes() {
            let attr = attr.map_err(|_| Unhandled("attribute syntax"))?;
            attribute_name(reader, &attr)?;
            let value = attr_value(&attr.value)?;
            match attr.key.as_ref() {
                b"xml:lang" => attrs.language = Some(value),
                b"xml:base" => attrs.base = Some(value),
                _ => {
                    let (resolved, local) = reader.resolve_attribute(attr.key);
                    let is_rdf = matches!(resolved, ResolveResult::Bound(ns) if ns.as_ref() == RDF);
                    if is_rdf && local.as_ref() == b"about" {
                        attrs.rdf_about = Some(value);
                    }
                }
            }
        }
        self.item_attrs = attrs;
        self.item_depth = self.depth;
        self.item_start = pos;
        self.frames.clear();
        self.frames.push(Frame::new(Role::Item));
        self.media = MediaAcc::default();
        self.acc = Some(match self.kind {
            Some(FeedKind::Atom) => ItemAcc::Atom(AtomAcc::default()),
            _ => ItemAcc::Rss(RssAcc::default()),
        });
        Ok(())
    }

    fn text(&mut self, raw: &[u8], is_cdata: bool) -> Result<(), Unhandled> {
        if self.depth == 0 && (is_cdata || !is_blank(raw)) {
            return Err(Unhandled("text outside the root element"));
        }
        match self.frames.last_mut() {
            Some(top) if self.item_depth != 0 && top.wants_text && top.text_open => {
                if is_cdata {
                    push_cdata(&mut top.text, raw)
                } else {
                    push_text(&mut top.text, raw)
                }
            }
            _ if is_cdata => Ok(()),
            _ => check_text(raw),
        }
    }

    fn doctype(&mut self, raw: &[u8]) -> Result<(), Unhandled> {
        if self.kind.is_some() || self.doctype_seen || raw.contains(&b'[') {
            return Err(Unhandled("doctype"));
        }
        self.doctype_seen = true;
        Ok(())
    }

    /// A comment or processing instruction ends the leading text.
    fn close_text(&mut self) {
        if self.item_depth != 0 {
            if let Some(top) = self.frames.last_mut() {
                top.text_open = false;
            }
        }
    }

    fn end<E>(
        &mut self,
        pos: usize,
        date_fn: &mut DateFn<E>,
        unescape: &mut UnescapeFn<E>,
    ) -> Result<(), Stop<E>> {
        if self.item_depth == 0 {
            if self.depth == 2 {
                self.in_channel = false;
            }
        } else if self.depth == self.item_depth {
            self.end_item(pos, date_fn, unescape)?;
        } else {
            self.end_in_item();
        }
        self.depth -= 1;
        self.root_closed = self.depth == 0;
        Ok(())
    }

    fn finish(self, data: &[u8]) -> Result<Document, Unhandled> {
        if self.depth != 0 {
            return Err(Unhandled("unclosed element"));
        }
        let kind = self.kind.ok_or(Unhandled("no root element"))?;
        if kind == FeedKind::Rss && self.entries.is_empty() {
            return Err(Unhandled("no items in channel"));
        }
        let mut header =
            Vec::with_capacity(data.len() - self.ranges.iter().map(|(a, b)| b - a).sum::<usize>());
        let mut cursor = 0;
        for (start, end) in &self.ranges {
            header.extend_from_slice(&data[cursor..*start]);
            cursor = *end;
        }
        header.extend_from_slice(&data[cursor..]);
        Ok(Document {
            kind,
            atom_ns: self.feed_ns.atom_uri(),
            header,
            entries: self.entries,
        })
    }
}

fn is_blank(raw: &[u8]) -> bool {
    raw.iter()
        .all(|b| matches!(b, b' ' | b'\t' | b'\r' | b'\n'))
}

/// An element, CDATA section or non-blank text: not allowed after the root.
fn is_trailing_content(event: &Event) -> bool {
    match event {
        Event::Start(_) | Event::CData(_) => true,
        Event::Text(text) => !is_blank(text),
        _ => false,
    }
}

/// Extract every entry of a well-formed UTF-8 RSS or Atom document.
pub fn parse<E>(
    data: &[u8],
    opts: &Options,
    date_fn: &mut DateFn<E>,
    unescape: &mut UnescapeFn<E>,
) -> Result<Document, Stop<E>> {
    validate::document(data)?;
    let mut reader = NsReader::from_reader(data);
    reader.config_mut().expand_empty_elements = true;
    let mut state = State {
        opts,
        depth: 0,
        kind: None,
        feed_ns: Ns::None,
        root_closed: false,
        doctype_seen: false,
        channel_seen: false,
        in_channel: false,
        item_depth: 0,
        item_start: 0,
        item_attrs: ItemAttrs::default(),
        frames: Vec::new(),
        acc: None,
        media: MediaAcc::default(),
        ranges: Vec::new(),
        entries: Vec::new(),
    };
    loop {
        let pos = reader.buffer_position() as usize;
        let (resolved, event) = reader
            .read_resolved_event()
            .map_err(|_| Unhandled("xml syntax"))?;
        let ns = Ns::from_resolved(&resolved);
        if state.root_closed && is_trailing_content(&event) {
            // libxml2 stops at content after the root element and keeps the tree.
            break;
        }
        match event {
            Event::Start(e) => state.start(ns, &e, pos, &reader)?,
            Event::End(_) => state.end(reader.buffer_position() as usize, date_fn, unescape)?,
            Event::Text(t) => state.text(&t, false)?,
            Event::CData(t) => state.text(&t, true)?,
            Event::Comment(c) => {
                validate::comment(&c)?;
                state.close_text();
            }
            Event::PI(pi) => {
                validate::processing_instruction(&pi)?;
                state.close_text();
            }
            Event::Decl(_) if pos != 0 => return Err(Unhandled("misplaced xml declaration").into()),
            Event::Decl(decl) => validate::declaration(&decl)?,
            Event::DocType(doctype) => state.doctype(&doctype)?,
            Event::Empty(_) => {}
            Event::Eof => break,
        }
    }
    Ok(state.finish(data)?)
}
