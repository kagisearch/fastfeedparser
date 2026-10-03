//! Port of `_parse_rss_feed_entry_fast`.
use crate::common::{
    content_from_element, fill_description, is_http_url, parse_date, populate_links, strip_shared,
    stripped,
};
use crate::model::{
    ContentEl, ContentOut, DateFn, EnclosureOut, Entry, ItemAttrs, LinkAttrs, MediaOut, Options,
    Stop, TagOut, Text, UnescapeFn,
};
use crate::ns::Ns;
use crate::text::py_strip;

/// Lowercased local names whose first occurrence among an item's children is
/// recorded, whatever the namespace.
const TRACKED: [&str; 15] = [
    "guid",
    "title",
    "description",
    "summary",
    "link",
    "pubdate",
    "published",
    "issued",
    "date",
    "lastbuilddate",
    "updated",
    "modified",
    "author",
    "creator",
    "comments",
];
const GUID: usize = 0;
const TITLE: usize = 1;
const DESCRIPTION: usize = 2;
const SUMMARY: usize = 3;
const LINK: usize = 4;
const PUBDATE: usize = 5;
const DATE: usize = 8;
const LASTBUILDDATE: usize = 9;
const MODIFIED: usize = 11;
const AUTHOR: usize = 12;
const CREATOR: usize = 13;
const COMMENTS: usize = 14;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum RssKind {
    Other,
    AtomLink,
    AtomId,
    Guid,
    Encoded,
    Content,
    Description,
    Enclosure,
    Category,
    Subject,
    AtomAuthor,
}

impl RssKind {
    /// Whether the element's text is used by `on_child`.
    pub fn uses_text(self) -> bool {
        !matches!(
            self,
            RssKind::Other | RssKind::AtomLink | RssKind::Enclosure | RssKind::AtomAuthor
        )
    }
}

/// The tracked-name slot and kind of an item child, like `_classify_rss_tag`.
pub fn classify(ns: Ns, local: &[u8]) -> (Option<usize>, RssKind) {
    let slot = TRACKED
        .iter()
        .position(|name| local.eq_ignore_ascii_case(name.as_bytes()));
    let kind = match (ns, local) {
        (Ns::AtomHttp, b"link") => RssKind::AtomLink,
        (Ns::AtomHttp, b"id") => RssKind::AtomId,
        (Ns::None, b"guid") => RssKind::Guid,
        (Ns::Content, b"encoded") => RssKind::Encoded,
        (Ns::None, b"content") => RssKind::Content,
        (Ns::None, b"description") => RssKind::Description,
        (Ns::None, b"enclosure") => RssKind::Enclosure,
        _ if local.eq_ignore_ascii_case(b"category") => RssKind::Category,
        (Ns::Dc, b"subject") => RssKind::Subject,
        (Ns::AtomHttp, b"author") => RssKind::AtomAuthor,
        _ => RssKind::Other,
    };
    (slot, kind)
}

/// Attributes read when an item child starts and used when it ends.
pub enum Extra {
    None,
    Domain(Option<String>),
    PermaLink(bool),
    Content {
        typ: Option<String>,
        language: Option<String>,
        base: Option<String>,
    },
}

struct GuidEl {
    text: Option<Text>,
    is_permalink: bool,
}

#[derive(Default)]
pub struct RssAcc {
    seen: [bool; 15],
    texts: [Option<Text>; 15],
    atom_id: Option<Text>,
    pub atom_links: Vec<LinkAttrs>,
    guid_el: Option<GuidEl>,
    encoded: Option<ContentEl>,
    raw_content: Option<ContentEl>,
    description: Option<Text>,
    categories: Vec<TagOut>,
    subjects: Vec<TagOut>,
    pub enclosures: Vec<EnclosureOut>,
    has_atom_author: bool,
    /// Text of the first atom:name under any atom:author child.
    atom_author_name: Option<Option<Text>>,
}

impl RssAcc {
    pub fn wants_text(&self, slot: Option<usize>, kind: RssKind) -> bool {
        slot.is_some_and(|s| !self.seen[s]) || kind.uses_text()
    }

    pub fn wants_author_name(&self) -> bool {
        self.atom_author_name.is_none()
    }

    /// One iteration of the Python child loop, run when the child ends.
    pub fn on_child(
        &mut self,
        slot: Option<usize>,
        kind: RssKind,
        text: Option<Text>,
        extra: Extra,
        first_name: Option<Option<Text>>,
        opts: &Options,
    ) {
        if let Some(s) = slot.filter(|s| !self.seen[*s]) {
            self.seen[s] = true;
            self.texts[s] = text.clone();
        }
        match (kind, extra) {
            (RssKind::Category, Extra::Domain(domain)) if opts.include_tags => {
                if let Some(term) = term_of(&text) {
                    self.categories.push(TagOut {
                        term,
                        scheme: domain,
                        label: None,
                    });
                }
            }
            (RssKind::Subject, _) if opts.include_tags => {
                if let Some(term) = term_of(&text) {
                    self.subjects.push(TagOut {
                        term,
                        scheme: None,
                        label: None,
                    });
                }
            }
            (RssKind::AtomId, _) if self.atom_id.is_none() => self.atom_id = text,
            (RssKind::Guid, Extra::PermaLink(is_permalink)) if self.guid_el.is_none() => {
                self.guid_el = Some(GuidEl { text, is_permalink });
            }
            (
                RssKind::Encoded,
                Extra::Content {
                    typ,
                    language,
                    base,
                },
            ) if self.encoded.is_none() => {
                self.encoded = Some(ContentEl {
                    text,
                    typ,
                    language,
                    base,
                });
            }
            (
                RssKind::Content,
                Extra::Content {
                    typ,
                    language,
                    base,
                },
            ) if self.raw_content.is_none() => {
                self.raw_content = Some(ContentEl {
                    text,
                    typ,
                    language,
                    base,
                });
            }
            (RssKind::Description, _) if self.description.is_none() => self.description = text,
            (RssKind::AtomAuthor, _) => {
                self.has_atom_author = true;
                if self.atom_author_name.is_none() {
                    self.atom_author_name = first_name;
                }
            }
            _ => {}
        }
    }

    fn text(&self, slot: usize) -> Option<&Text> {
        self.texts[slot].as_ref()
    }

    /// First non-empty text among consecutive slots `from..=to`.
    fn first_text(&self, from: usize, to: usize) -> Option<&Text> {
        (from..=to).find_map(|s| self.text(s))
    }

    /// The entry-building half of the Python function.
    pub fn finish<E>(
        mut self,
        item: ItemAttrs,
        media: Vec<MediaOut>,
        opts: &Options,
        date_fn: &mut DateFn<E>,
        unescape: &mut UnescapeFn<E>,
    ) -> Result<Entry, Stop<E>> {
        let mut entry = Entry::default();
        let rss_guid = self.text(GUID).cloned();
        let rdf_about = item.rdf_about.as_deref().filter(|a| !a.is_empty());
        entry.id = self
            .atom_id
            .as_deref()
            .map(String::as_str)
            .or(rss_guid.as_deref().map(String::as_str))
            .or(rdf_about)
            .map(|id| py_strip(id).to_owned());
        entry.title = self.text(TITLE).map(|t| py_strip(t).to_owned());
        entry.description = self.first_text(DESCRIPTION, SUMMARY).map(strip_shared);
        entry.link = self.text(LINK).map(|t| py_strip(t).to_owned());

        if let Some(source) = self.first_text(PUBDATE, DATE) {
            entry.published = parse_date(source, date_fn)?;
        }
        if let Some(source) = self.first_text(LASTBUILDDATE, MODIFIED) {
            entry.updated = parse_date(source, date_fn)?;
        }
        if let (None, Some(guid)) = (&entry.published, &rss_guid) {
            if !is_http_url(guid) {
                entry.published = parse_date(guid, date_fn)?;
            }
        }
        if entry.published.is_none() {
            entry.published = entry.updated.clone();
        }

        if self.atom_links.is_empty() {
            if let (None, Some(guid)) = (&entry.link, &rss_guid) {
                if is_http_url(guid) {
                    entry.link = Some(guid.to_string());
                }
            }
        } else {
            let guid_text = self
                .guid_el
                .as_ref()
                .and_then(|g| g.text.as_deref())
                .map(|t| py_strip(t));
            let is_permalink = self.guid_el.as_ref().is_some_and(|g| g.is_permalink);
            let links = std::mem::take(&mut self.atom_links);
            entry.links = populate_links(&mut entry.link, links, guid_text, is_permalink)?;
        }
        if entry.id.is_none() {
            entry.id = entry.link.clone();
        }

        if opts.include_content {
            entry.content = self.content(&item)?;
            fill_description(&mut entry, unescape)?;
        }
        entry.media = media;
        entry.enclosures = std::mem::take(&mut self.enclosures);
        entry.author = self.author();
        entry.comments = self.text(COMMENTS).map(|t| py_strip(t).to_owned());
        entry.tags = std::mem::take(&mut self.categories);
        entry.tags.append(&mut self.subjects);
        Ok(entry)
    }

    fn content(&mut self, item: &ItemAttrs) -> Result<Option<ContentOut>, crate::model::Unhandled> {
        if let Some(el) = self.encoded.take().or_else(|| self.raw_content.take()) {
            return content_from_element(el).map(Some);
        }
        Ok(self.description.take().map(|value| ContentOut {
            typ: "text/html".to_owned(),
            language: item.language.clone(),
            base: item.base.clone(),
            value,
        }))
    }

    fn author(&self) -> Option<String> {
        if let Some(raw) = self.first_text(AUTHOR, CREATOR) {
            return Some(py_strip(raw).to_owned());
        }
        if !self.has_atom_author {
            return None;
        }
        let name = self.atom_author_name.as_ref()?;
        stripped(name).filter(|n| !n.is_empty())
    }
}

fn term_of(text: &Option<Text>) -> Option<String> {
    stripped(text).filter(|t| !t.is_empty())
}
