//! Handling of the elements inside one item or entry.
use std::rc::Rc;

use quick_xml::events::BytesStart;

use crate::atom::{self, AtomAcc, AtomKind};
use crate::common::enclosure;
use crate::extract::{Reader, State};
use crate::media::{FirstCredit, FirstDesc, Thumb};
use crate::model::{
    ContentEl, DateFn, FeedKind, LinkAttrs, MediaOut, Stop, TagOut, Text, UnescapeFn, Unhandled,
};
use crate::ns::Ns;
use crate::rss::{self, Extra, RssAcc, RssKind};
use crate::text::py_int;
use crate::validate::read_attrs;

pub(crate) enum Role {
    Item,
    Rss { slot: Option<usize>, kind: RssKind },
    Atom(AtomKind),
    Nested,
}

#[derive(Clone, Copy)]
pub(crate) enum MediaRole {
    None,
    Content(usize),
    Title,
    Text,
    Description,
    Credit,
}

/// One open element inside an item.
pub(crate) struct Frame {
    pub(crate) role: Role,
    pub(crate) media: MediaRole,
    pub(crate) is_author_name: bool,
    pub(crate) extra: Extra,
    pub(crate) credit_scheme: Option<String>,
    pub(crate) wants_text: bool,
    /// False once a child node has been seen: only leading text counts,
    /// like lxml's `.text`.
    pub(crate) text_open: bool,
    pub(crate) text: String,
    pub(crate) first_desc: FirstDesc,
    pub(crate) first_credit: FirstCredit,
    pub(crate) first_title: Option<Option<Text>>,
    pub(crate) first_text: Option<Option<Text>>,
    pub(crate) first_thumb: Option<Option<String>>,
    /// media:content children still missing a description or credit.
    pub(crate) pending_media: Vec<usize>,
    /// Text of the first atom:name child, for an atom:author element.
    pub(crate) first_name: Option<Option<Text>>,
}

impl Frame {
    pub(crate) fn new(role: Role) -> Frame {
        Frame {
            role,
            media: MediaRole::None,
            is_author_name: false,
            extra: Extra::None,
            credit_scheme: None,
            wants_text: false,
            text_open: false,
            text: String::new(),
            first_desc: None,
            first_credit: None,
            first_title: None,
            first_text: None,
            first_thumb: None,
            pending_media: Vec::new(),
            first_name: None,
        }
    }

    fn is_atom_author(&self) -> bool {
        matches!(
            self.role,
            Role::Rss {
                kind: RssKind::AtomAuthor,
                ..
            } | Role::Atom(AtomKind::Author)
        )
    }
}

pub(crate) enum ItemAcc {
    Rss(RssAcc),
    Atom(AtomAcc),
}

fn int_attr(value: Option<String>) -> Result<Option<i64>, Unhandled> {
    match value {
        Some(v) => py_int(&v),
        None => Ok(None),
    }
}

fn link_attrs(reader: &Reader, e: &BytesStart) -> Result<LinkAttrs, Unhandled> {
    let [rel, typ, href, link, title] = read_attrs(
        reader,
        e,
        [
            &b"rel"[..],
            &b"type"[..],
            &b"href"[..],
            &b"link"[..],
            &b"title"[..],
        ],
    )?;
    Ok(LinkAttrs {
        rel,
        typ,
        href,
        link,
        title,
    })
}

fn content_extra(reader: &Reader, e: &BytesStart) -> Result<Extra, Unhandled> {
    let [typ, language, base] = read_attrs(
        reader,
        e,
        [&b"type"[..], &b"xml:lang"[..], &b"xml:base"[..]],
    )?;
    Ok(Extra::Content {
        typ,
        language,
        base,
    })
}

impl State<'_> {
    pub(crate) fn start_in_item(
        &mut self,
        ns: Ns,
        local: &[u8],
        e: &BytesStart,
        reader: &Reader,
    ) -> Result<(), Unhandled> {
        let parent = self.frames.last_mut().expect("item frame");
        parent.text_open = false;
        let parent_is_author = parent.is_atom_author();
        let parent_has_name = parent.first_name.is_some();
        let parent_is_media_content = matches!(parent.media, MediaRole::Content(_));

        let mut frame = if self.depth == self.item_depth + 1 {
            self.start_child(ns, local, e, reader)?
        } else {
            read_attrs(reader, e, [])?;
            Frame::new(Role::Nested)
        };

        let atom_ns = if self.kind == Some(FeedKind::Atom) {
            self.feed_ns
        } else {
            Ns::AtomHttp
        };
        if parent_is_author && !parent_has_name && ns == atom_ns && local == b"name" {
            frame.is_author_name = true;
            frame.wants_text |= match self.acc.as_ref().expect("item accumulator") {
                ItemAcc::Rss(acc) => acc.wants_author_name(),
                ItemAcc::Atom(acc) => acc.wants_author_name(),
            };
        }
        if ns == Ns::Media && self.opts.include_media {
            self.start_media(local, e, reader, &mut frame, parent_is_media_content)?;
        }
        frame.text_open = frame.wants_text;
        self.frames.push(frame);
        Ok(())
    }

    /// A direct child of the item: classify it and read what its start tag carries.
    fn start_child(
        &mut self,
        ns: Ns,
        local: &[u8],
        e: &BytesStart,
        reader: &Reader,
    ) -> Result<Frame, Unhandled> {
        let opts = self.opts;
        match self.acc.as_mut().expect("item accumulator") {
            ItemAcc::Rss(acc) => {
                let (slot, kind) = rss::classify(ns, local);
                let mut frame = Frame::new(Role::Rss { slot, kind });
                frame.wants_text = acc.wants_text(slot, kind);
                match kind {
                    RssKind::AtomLink => acc.atom_links.push(link_attrs(reader, e)?),
                    RssKind::Enclosure => {
                        let [url, typ, length] =
                            read_attrs(reader, e, [&b"url"[..], &b"type"[..], &b"length"[..]])?;
                        if opts.include_enclosures {
                            acc.enclosures.extend(enclosure(url, typ, length)?);
                        }
                    }
                    RssKind::Category => {
                        let [domain] = read_attrs(reader, e, [&b"domain"[..]])?;
                        frame.extra = Extra::Domain(domain);
                    }
                    RssKind::Guid => {
                        let [permalink] = read_attrs(reader, e, [&b"isPermaLink"[..]])?;
                        frame.extra = Extra::PermaLink(permalink.as_deref() == Some("true"));
                    }
                    RssKind::Encoded | RssKind::Content => frame.extra = content_extra(reader, e)?,
                    _ => {
                        read_attrs(reader, e, [])?;
                    }
                }
                Ok(frame)
            }
            ItemAcc::Atom(acc) => {
                if ns == self.feed_ns && local == b"entry" {
                    return Err(Unhandled("nested entry"));
                }
                let kind = atom::classify(self.feed_ns, ns, local);
                let mut frame = Frame::new(Role::Atom(kind));
                frame.wants_text = kind.uses_text();
                match kind {
                    AtomKind::Link => acc.on_link(link_attrs(reader, e)?),
                    AtomKind::Category => {
                        let [term, scheme, label] =
                            read_attrs(reader, e, [&b"term"[..], &b"scheme"[..], &b"label"[..]])?;
                        if let (true, Some(term)) =
                            (opts.include_tags, term.filter(|t| !t.is_empty()))
                        {
                            acc.categories.push(TagOut {
                                term,
                                scheme,
                                label,
                            });
                        }
                    }
                    AtomKind::Enclosure => {
                        let [url, typ, length] =
                            read_attrs(reader, e, [&b"url"[..], &b"type"[..], &b"length"[..]])?;
                        if opts.include_enclosures {
                            acc.enclosures.extend(enclosure(url, typ, length)?);
                        }
                    }
                    AtomKind::Content => {
                        frame.extra = content_extra(reader, e)?;
                        frame.wants_text = acc.wants_content(opts);
                    }
                    _ => {
                        read_attrs(reader, e, [])?;
                    }
                }
                Ok(frame)
            }
        }
    }

    fn start_media(
        &mut self,
        local: &[u8],
        e: &BytesStart,
        reader: &Reader,
        frame: &mut Frame,
        parent_is_media_content: bool,
    ) -> Result<(), Unhandled> {
        match local {
            b"content" => {
                let [url, typ, medium, width, height] = read_attrs(
                    reader,
                    e,
                    [
                        &b"url"[..],
                        &b"type"[..],
                        &b"medium"[..],
                        &b"width"[..],
                        &b"height"[..],
                    ],
                )?;
                let out = MediaOut {
                    url,
                    typ,
                    medium,
                    width: int_attr(width)?,
                    height: int_attr(height)?,
                    ..MediaOut::default()
                };
                frame.media = MediaRole::Content(self.media.begin(out));
            }
            b"title" => frame.media = MediaRole::Title,
            b"text" => frame.media = MediaRole::Text,
            b"description" => frame.media = MediaRole::Description,
            b"credit" => {
                let [scheme] = read_attrs(reader, e, [&b"scheme"[..]])?;
                frame.credit_scheme = scheme;
                frame.media = MediaRole::Credit;
            }
            b"thumbnail" => {
                let [url, width, height] =
                    read_attrs(reader, e, [&b"url"[..], &b"width"[..], &b"height"[..]])?;
                let parent = self.frames.last_mut().expect("item frame");
                if !parent_is_media_content {
                    self.media.add_thumb(Thumb {
                        url,
                        width: int_attr(width)?,
                        height: int_attr(height)?,
                    });
                } else if parent.first_thumb.is_none() {
                    parent.first_thumb = Some(url);
                }
            }
            _ => {}
        }
        frame.wants_text |= !matches!(frame.media, MediaRole::None | MediaRole::Content(_));
        Ok(())
    }

    pub(crate) fn end_in_item(&mut self) {
        let mut frame = self.frames.pop().expect("open frame");
        let has_text = frame.wants_text && !frame.text.is_empty();
        let text: Option<Text> = has_text.then(|| Rc::new(std::mem::take(&mut frame.text)));
        for idx in frame.pending_media.drain(..) {
            self.media.fill(idx, &frame.first_desc, &frame.first_credit);
        }
        let parent = self.frames.last_mut().expect("parent frame");
        match frame.media {
            MediaRole::Content(idx) => {
                self.media.set_title(idx, &frame.first_title);
                self.media.set_text(idx, &frame.first_text);
                self.media.set_thumbnail(idx, frame.first_thumb.take());
                self.media.fill(idx, &frame.first_desc, &frame.first_credit);
                if self.media.needs_parent(idx) {
                    parent.pending_media.push(idx);
                }
            }
            MediaRole::Title if parent.first_title.is_none() => {
                parent.first_title = Some(text.clone())
            }
            MediaRole::Text if parent.first_text.is_none() => {
                parent.first_text = Some(text.clone())
            }
            MediaRole::Description if parent.first_desc.is_none() => {
                parent.first_desc = Some(text.clone())
            }
            MediaRole::Credit if parent.first_credit.is_none() => {
                parent.first_credit = Some((text.clone(), frame.credit_scheme.take()));
            }
            _ => {}
        }
        if frame.is_author_name && parent.first_name.is_none() {
            parent.first_name = Some(text.clone());
        }
        let opts = self.opts;
        match (self.acc.as_mut().expect("item accumulator"), frame.role) {
            (ItemAcc::Rss(acc), Role::Rss { slot, kind }) => {
                acc.on_child(slot, kind, text, frame.extra, frame.first_name, opts);
            }
            (ItemAcc::Atom(acc), Role::Atom(AtomKind::Author)) => acc.on_author(frame.first_name),
            (ItemAcc::Atom(acc), Role::Atom(AtomKind::Content)) => {
                if let Extra::Content {
                    typ,
                    language,
                    base,
                } = frame.extra
                {
                    acc.on_content(
                        ContentEl {
                            text,
                            typ,
                            language,
                            base,
                        },
                        opts,
                    );
                }
            }
            (ItemAcc::Atom(acc), Role::Atom(kind)) => acc.on_text_child(kind, text),
            _ => {}
        }
    }

    pub(crate) fn end_item<E>(
        &mut self,
        pos: usize,
        date_fn: &mut DateFn<E>,
        unescape: &mut UnescapeFn<E>,
    ) -> Result<(), Stop<E>> {
        let mut frame = self.frames.pop().expect("item frame");
        for idx in frame.pending_media.drain(..) {
            self.media.fill(idx, &frame.first_desc, &frame.first_credit);
        }
        let media = std::mem::take(&mut self.media).finish();
        let entry = match self.acc.take().expect("item accumulator") {
            ItemAcc::Rss(acc) => acc.finish(
                std::mem::take(&mut self.item_attrs),
                media,
                self.opts,
                date_fn,
                unescape,
            )?,
            ItemAcc::Atom(acc) => acc.finish(media, date_fn, unescape)?,
        };
        self.entries.push(entry);
        self.ranges.push((self.item_start, pos));
        self.item_depth = 0;
        Ok(())
    }
}
