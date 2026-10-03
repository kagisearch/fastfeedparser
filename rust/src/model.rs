//! Data passed between the extractor, the entry builders and the bindings.
use std::rc::Rc;

/// Element text. Shared because one text can feed several fields.
pub type Text = Rc<String>;

/// The document is outside what the extractor reproduces exactly. The caller
/// parses it with the lxml path instead.
#[derive(Debug, PartialEq)]
pub struct Unhandled(pub &'static str);

/// Why extraction stopped early.
pub enum Stop<E> {
    Unhandled(Unhandled),
    /// The Python date callback raised.
    Callback(E),
}

impl<E> From<Unhandled> for Stop<E> {
    fn from(u: Unhandled) -> Self {
        Stop::Unhandled(u)
    }
}

/// Parses a date the fast paths could not decide.
pub type DateFn<'a, E> = dyn FnMut(&str) -> Result<Option<String>, E> + 'a;

#[derive(Clone, Copy)]
pub struct Options {
    pub include_content: bool,
    pub include_tags: bool,
    pub include_media: bool,
    pub include_enclosures: bool,
}

#[derive(Default)]
pub struct LinkAttrs {
    pub rel: Option<String>,
    pub typ: Option<String>,
    pub href: Option<String>,
    pub link: Option<String>,
    pub title: Option<String>,
}

pub struct ContentEl {
    pub text: Option<Text>,
    pub typ: Option<String>,
    pub language: Option<String>,
    pub base: Option<String>,
}

#[derive(Default)]
pub struct ItemAttrs {
    pub language: Option<String>,
    pub base: Option<String>,
    pub rdf_about: Option<String>,
}

pub struct LinkOut {
    pub rel: Option<String>,
    pub typ: Option<String>,
    pub href: String,
    pub title: Option<String>,
    /// The alternate link built from a guid has no title key at all.
    pub has_title: bool,
}

pub struct ContentOut {
    pub typ: String,
    pub language: Option<String>,
    pub base: Option<String>,
    pub value: Text,
}

pub struct TagOut {
    pub term: String,
    pub scheme: Option<String>,
    pub label: Option<String>,
}

pub enum Length {
    Absent,
    /// An empty length attribute is kept as an empty string.
    Empty,
    Int(i64),
}

pub struct EnclosureOut {
    pub url: String,
    pub typ: Option<String>,
    pub length: Length,
}

#[derive(Default)]
pub struct MediaOut {
    pub url: Option<String>,
    pub typ: Option<String>,
    pub medium: Option<String>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub title: Option<String>,
    pub text: Option<String>,
    pub description: Option<String>,
    pub credit: Option<String>,
    pub credit_scheme: Option<String>,
    pub thumbnail_url: Option<String>,
}

#[derive(Default)]
pub struct Entry {
    pub id: Option<String>,
    pub title: Option<String>,
    /// May share its allocation with the content value.
    pub description: Option<Text>,
    pub link: Option<String>,
    pub published: Option<String>,
    pub updated: Option<String>,
    pub links: Vec<LinkOut>,
    pub content: Option<ContentOut>,
    pub media: Vec<MediaOut>,
    pub enclosures: Vec<EnclosureOut>,
    pub author: Option<String>,
    pub comments: Option<String>,
    pub tags: Vec<TagOut>,
}

#[derive(Clone, Copy, PartialEq)]
pub enum FeedKind {
    Rss,
    Atom,
}

pub struct Document {
    pub kind: FeedKind,
    pub atom_ns: Option<&'static str>,
    /// The input with every item or entry element cut out.
    pub header: Vec<u8>,
    pub entries: Vec<Entry>,
}
