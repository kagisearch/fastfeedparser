//! Port of `_parse_atom_feed_entry_fast`.
use std::rc::Rc;

use crate::common::{content_from_element, parse_date, populate_links};
use crate::model::{
    ContentEl, DateFn, EnclosureOut, Entry, LinkAttrs, MediaOut, Options, Stop, TagOut, Text,
};
use crate::ns::Ns;
use crate::text::py_strip;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum AtomKind {
    Other,
    Id,
    Title,
    Summary,
    Published,
    Updated,
    PublishedFallback,
    UpdatedFallback,
    Link,
    Content,
    Author,
    Category,
    Enclosure,
}

impl AtomKind {
    pub fn uses_text(self) -> bool {
        !matches!(
            self,
            AtomKind::Other
                | AtomKind::Link
                | AtomKind::Author
                | AtomKind::Category
                | AtomKind::Enclosure
        )
    }
}

/// The kind of an entry child, like `_atom_entry_tag_kinds`. Atom 0.3 names
/// its dates issued/modified; each version accepts the other's as fallbacks.
pub fn classify(feed_ns: Ns, ns: Ns, local: &[u8]) -> AtomKind {
    if ns == Ns::None && local == b"enclosure" {
        return AtomKind::Enclosure;
    }
    if ns != feed_ns {
        return AtomKind::Other;
    }
    let is_03 = feed_ns == Ns::Atom03;
    match local {
        b"id" => AtomKind::Id,
        b"title" => AtomKind::Title,
        b"summary" => AtomKind::Summary,
        b"link" => AtomKind::Link,
        b"content" => AtomKind::Content,
        b"author" => AtomKind::Author,
        b"category" => AtomKind::Category,
        b"published" if is_03 => AtomKind::PublishedFallback,
        b"published" => AtomKind::Published,
        b"issued" if is_03 => AtomKind::Published,
        b"issued" => AtomKind::PublishedFallback,
        b"updated" if is_03 => AtomKind::UpdatedFallback,
        b"updated" => AtomKind::Updated,
        b"modified" if is_03 => AtomKind::Updated,
        b"modified" => AtomKind::UpdatedFallback,
        _ => AtomKind::Other,
    }
}

#[derive(Default)]
pub struct AtomAcc {
    id: Option<String>,
    title: Option<String>,
    summary: Option<String>,
    published: Option<Text>,
    updated: Option<Text>,
    published_fallback: Option<Text>,
    updated_fallback: Option<Text>,
    first_link_href: Option<String>,
    atom_links: Vec<LinkAttrs>,
    pub categories: Vec<TagOut>,
    pub enclosures: Vec<EnclosureOut>,
    content: Option<ContentEl>,
    /// Stripped name of the first author whose first atom:name has text.
    author_name: Option<String>,
}

impl AtomAcc {
    pub fn wants_content(&self, opts: &Options) -> bool {
        opts.include_content && self.content.is_none()
    }

    pub fn wants_author_name(&self) -> bool {
        self.author_name.is_none()
    }

    pub fn on_link(&mut self, attrs: LinkAttrs) {
        if self.first_link_href.is_none() {
            if let Some(href) = attrs.href.as_deref().filter(|h| !h.is_empty()) {
                self.first_link_href = Some(py_strip(href).to_owned());
            }
        }
        self.atom_links.push(attrs);
    }

    /// One iteration of the Python child loop for kinds that read text.
    pub fn on_text_child(&mut self, kind: AtomKind, text: Option<Text>) {
        let Some(text) = text else {
            return;
        };
        match kind {
            AtomKind::Id if self.id.is_none() => self.id = Some(py_strip(&text).to_owned()),
            AtomKind::Title if self.title.is_none() => {
                self.title = Some(py_strip(&text).to_owned())
            }
            AtomKind::Summary if self.summary.is_none() => {
                self.summary = Some(py_strip(&text).to_owned())
            }
            AtomKind::Published if self.published.is_none() => self.published = Some(text),
            AtomKind::Updated if self.updated.is_none() => self.updated = Some(text),
            AtomKind::PublishedFallback if self.published_fallback.is_none() => {
                self.published_fallback = Some(text)
            }
            AtomKind::UpdatedFallback if self.updated_fallback.is_none() => {
                self.updated_fallback = Some(text)
            }
            _ => {}
        }
    }

    pub fn on_content(&mut self, el: ContentEl, opts: &Options) {
        if self.wants_content(opts) {
            self.content = Some(el);
        }
    }

    /// `first_name` is the text of the author's first atom:name child.
    pub fn on_author(&mut self, first_name: Option<Option<Text>>) {
        if self.author_name.is_none() {
            if let Some(Some(name)) = first_name {
                self.author_name = Some(py_strip(&name).to_owned());
            }
        }
    }

    /// The entry-building half of the Python function.
    pub fn finish<E>(
        self,
        media: Vec<MediaOut>,
        date_fn: &mut DateFn<E>,
    ) -> Result<Entry, Stop<E>> {
        let mut entry = Entry {
            id: self.id,
            title: self.title,
            description: self.summary.map(Rc::new),
            link: self.first_link_href.filter(|href| !href.is_empty()),
            ..Entry::default()
        };
        if let Some(source) = &self.published {
            entry.published = parse_date(source, date_fn)?;
        }
        if let Some(source) = &self.updated {
            entry.updated = parse_date(source, date_fn)?;
        }
        if let (None, Some(source)) = (&entry.published, &self.published_fallback) {
            entry.published = parse_date(source, date_fn)?;
        }
        if let (None, Some(source)) = (&entry.updated, &self.updated_fallback) {
            entry.updated = parse_date(source, date_fn)?;
        }
        if entry.published.is_none() {
            entry.published = entry.updated.clone();
        }

        entry.links = populate_links(&mut entry.link, self.atom_links, None, false)?;
        if entry.id.is_none() {
            entry.id = entry.link.clone();
        }
        if let Some(el) = self.content {
            entry.content = Some(content_from_element(el)?);
        }
        entry.media = media;
        entry.enclosures = self.enclosures;
        entry.author = self.author_name.filter(|n| !n.is_empty());
        entry.tags = self.categories;
        Ok(entry)
    }
}
