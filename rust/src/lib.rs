//! Native entry extractor for fastfeedparser.
//!
//! `parse_entries` handles well-formed UTF-8 RSS and Atom documents and
//! returns entries equal to what the lxml path produces. For anything else it
//! returns a reason string and the caller uses the lxml path.
mod atom;
mod common;
mod dates;
mod extract;
mod item;
mod media;
mod model;
mod ns;
mod rss;
mod text;
mod validate;

use std::rc::Rc;

use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString};

use dates::Fast;
pub use model::Unhandled;
use model::{Entry, FeedKind, Length, MediaOut, Options, Stop};

fn new_dict<'py>(py: Python<'py>) -> Bound<'py, PyDict> {
    PyDict::new(py)
}

fn links_to_py<'py>(py: Python<'py>, entry: &Entry) -> PyResult<Bound<'py, PyList>> {
    let links = PyList::empty(py);
    for link in &entry.links {
        let d = new_dict(py);
        d.set_item(intern!(py, "rel"), link.rel.as_deref())?;
        d.set_item(intern!(py, "type"), link.typ.as_deref())?;
        d.set_item(intern!(py, "href"), &link.href)?;
        if link.has_title {
            d.set_item(intern!(py, "title"), link.title.as_deref())?;
        }
        links.append(d)?;
    }
    Ok(links)
}

fn media_to_py<'py>(py: Python<'py>, media: &[MediaOut]) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for m in media {
        let d = new_dict(py);
        let texts = [
            (intern!(py, "url"), &m.url),
            (intern!(py, "type"), &m.typ),
            (intern!(py, "medium"), &m.medium),
        ];
        for (key, value) in texts {
            if let Some(v) = value {
                d.set_item(key, v)?;
            }
        }
        if let Some(v) = m.width {
            d.set_item(intern!(py, "width"), v)?;
        }
        if let Some(v) = m.height {
            d.set_item(intern!(py, "height"), v)?;
        }
        let texts = [
            (intern!(py, "title"), &m.title),
            (intern!(py, "text"), &m.text),
            (intern!(py, "description"), &m.description),
            (intern!(py, "credit"), &m.credit),
            (intern!(py, "credit_scheme"), &m.credit_scheme),
            (intern!(py, "thumbnail_url"), &m.thumbnail_url),
        ];
        for (key, value) in texts {
            if let Some(v) = value {
                d.set_item(key, v)?;
            }
        }
        list.append(d)?;
    }
    Ok(list)
}

fn enclosures_to_py<'py>(py: Python<'py>, entry: &Entry) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for enc in &entry.enclosures {
        let d = new_dict(py);
        d.set_item(intern!(py, "url"), &enc.url)?;
        if let Some(typ) = &enc.typ {
            d.set_item(intern!(py, "type"), typ)?;
        }
        match enc.length {
            Length::Absent => {}
            Length::Empty => d.set_item(intern!(py, "length"), "")?,
            Length::Int(n) => d.set_item(intern!(py, "length"), n)?,
        }
        list.append(d)?;
    }
    Ok(list)
}

fn tags_to_py<'py>(py: Python<'py>, entry: &Entry) -> PyResult<Bound<'py, PyList>> {
    let list = PyList::empty(py);
    for tag in &entry.tags {
        let d = new_dict(py);
        d.set_item(intern!(py, "term"), &tag.term)?;
        d.set_item(intern!(py, "scheme"), tag.scheme.as_deref())?;
        d.set_item(intern!(py, "label"), tag.label.as_deref())?;
        list.append(d)?;
    }
    Ok(list)
}

/// Build the entry mapping. Keys follow the order the lxml path inserts them.
fn entry_to_py<'py>(
    py: Python<'py>,
    entry_cls: &Bound<'py, PyAny>,
    entry: &Entry,
) -> PyResult<Bound<'py, PyDict>> {
    let d = entry_cls.call0()?.downcast_into::<PyDict>()?;
    if let Some(id) = &entry.id {
        d.set_item(intern!(py, "id"), id)?;
    }
    d.set_item(intern!(py, "title"), entry.title.as_deref().unwrap_or(""))?;
    let description = entry
        .description
        .as_ref()
        .map(|text| PyString::new(py, text));
    if let Some(description) = &description {
        d.set_item(intern!(py, "description"), description)?;
    }
    if let Some(link) = &entry.link {
        d.set_item(intern!(py, "link"), link)?;
    }
    if let Some(published) = &entry.published {
        d.set_item(intern!(py, "published"), published)?;
    }
    if let Some(updated) = &entry.updated {
        d.set_item(intern!(py, "updated"), updated)?;
    }
    d.set_item(intern!(py, "links"), links_to_py(py, entry)?)?;
    if let Some(content) = &entry.content {
        let c = new_dict(py);
        c.set_item(intern!(py, "type"), &content.typ)?;
        c.set_item(intern!(py, "language"), content.language.as_deref())?;
        c.set_item(intern!(py, "base"), content.base.as_deref())?;
        // An RSS description that doubles as the content is one Python string.
        match (&entry.description, &description) {
            (Some(text), Some(shared)) if Rc::ptr_eq(text, &content.value) => {
                c.set_item(intern!(py, "value"), shared)?;
            }
            _ => c.set_item(intern!(py, "value"), content.value.as_str())?,
        }
        d.set_item(intern!(py, "content"), PyList::new(py, [c])?)?;
    }
    if !entry.media.is_empty() {
        d.set_item(intern!(py, "media_content"), media_to_py(py, &entry.media)?)?;
    }
    if !entry.enclosures.is_empty() {
        d.set_item(intern!(py, "enclosures"), enclosures_to_py(py, entry)?)?;
    }
    if let Some(author) = &entry.author {
        d.set_item(intern!(py, "author"), author)?;
    }
    if let Some(comments) = &entry.comments {
        d.set_item(intern!(py, "comments"), comments)?;
    }
    if !entry.tags.is_empty() {
        d.set_item(intern!(py, "tags"), tags_to_py(py, entry)?)?;
    }
    if let Some(author) = entry.author.as_deref().filter(|a| !a.is_empty()) {
        let detail = new_dict(py);
        detail.set_item(intern!(py, "name"), author)?;
        d.set_item(intern!(py, "author_detail"), &detail)?;
        d.set_item(intern!(py, "authors"), PyList::new(py, [detail])?)?;
    }
    Ok(d)
}

/// Extract the entries of an RSS or Atom document.
///
/// Returns `(kind, atom_namespace, header_bytes, entries)`, where
/// `header_bytes` is the document with its items removed, or a string naming
/// why the document must go to the lxml path. `entry_cls` is called with no
/// arguments to create each entry mapping; `parse_date` is called for dates
/// the built-in fast paths cannot decide.
#[pyfunction]
#[pyo3(signature = (data, entry_cls, parse_date, include_content=true, include_tags=true, include_media=true, include_enclosures=true))]
#[allow(clippy::too_many_arguments)]
fn parse_entries<'py>(
    py: Python<'py>,
    data: &[u8],
    entry_cls: &Bound<'py, PyAny>,
    parse_date: &Bound<'py, PyAny>,
    include_content: bool,
    include_tags: bool,
    include_media: bool,
    include_enclosures: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let opts = Options {
        include_content,
        include_tags,
        include_media,
        include_enclosures,
    };
    let mut date_fn =
        |raw: &str| -> PyResult<Option<String>> { parse_date.call1((raw,))?.extract() };
    let doc = match extract::parse(data, &opts, &mut date_fn) {
        Ok(doc) => doc,
        Err(Stop::Unhandled(Unhandled(reason))) => return Ok(PyString::new(py, reason).into_any()),
        Err(Stop::Callback(err)) => return Err(err),
    };
    let entries = PyList::empty(py);
    for entry in &doc.entries {
        entries.append(entry_to_py(py, entry_cls, entry)?)?;
    }
    let kind = match doc.kind {
        FeedKind::Rss => "rss",
        FeedKind::Atom => "atom",
    };
    let result = (kind, doc.atom_ns, PyBytes::new(py, &doc.header), entries).into_pyobject(py)?;
    Ok(result.into_any())
}

/// The built-in date fast path, exposed for parity tests.
///
/// Returns `(state, value)`: state 0 with the ISO string, 1 for "not a date",
/// 2 when the input is left to Python.
#[pyfunction]
fn fast_date(raw: &str) -> (u8, Option<String>) {
    match dates::fast_parse(raw) {
        Fast::Iso(iso) => (0, Some(iso)),
        Fast::NoDate => (1, None),
        Fast::Unknown => (2, None),
    }
}

/// Entry points for `examples/bench.rs`, which times the stages without Python.
#[doc(hidden)]
pub mod bench_api {
    use crate::model::{Options, Stop};

    pub fn validate(data: &[u8]) -> bool {
        crate::validate::document(data).is_ok()
    }

    /// Extract with every option on; returns the entry count if handled.
    pub fn extract(data: &[u8]) -> Option<usize> {
        let opts = Options {
            include_content: true,
            include_tags: true,
            include_media: true,
            include_enclosures: true,
        };
        let mut no_fallback = |_: &str| -> Result<Option<String>, ()> { Ok(None) };
        match crate::extract::parse(data, &opts, &mut no_fallback) {
            Ok(doc) => Some(doc.entries.len()),
            Err(Stop::Unhandled(_) | Stop::Callback(())) => None,
        }
    }
}

#[pymodule]
fn fastfeedparser_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_entries, m)?)?;
    m.add_function(wrap_pyfunction!(fast_date, m)?)?;
    Ok(())
}
