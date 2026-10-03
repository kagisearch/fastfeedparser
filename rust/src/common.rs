//! Entry-building steps shared by RSS and Atom.
use std::rc::Rc;

use crate::dates::{fast_parse, Fast};
use crate::model::{
    ContentEl, ContentOut, DateFn, EnclosureOut, Length, LinkAttrs, LinkOut, Stop, Text, Unhandled,
};
use crate::text::{py_int, py_strip};

pub fn is_http_url(s: &str) -> bool {
    s.starts_with("http://") || s.starts_with("https://")
}

/// `text.strip()` for a text Python treats as truthy.
pub fn stripped(text: &Option<Text>) -> Option<String> {
    text.as_ref().map(|t| py_strip(t).to_owned())
}

/// `text.strip()` that shares the allocation when there is nothing to strip.
pub fn strip_shared(text: &Text) -> Text {
    let stripped = py_strip(text);
    if stripped.len() == text.len() {
        Rc::clone(text)
    } else {
        Rc::new(stripped.to_owned())
    }
}

pub fn parse_date<E>(raw: &str, fallback: &mut DateFn<E>) -> Result<Option<String>, Stop<E>> {
    match fast_parse(raw) {
        Fast::Iso(iso) => Ok(Some(iso)),
        Fast::NoDate => Ok(None),
        Fast::Unknown => fallback(raw).map_err(Stop::Callback),
    }
}

pub fn enclosure(
    url: Option<String>,
    typ: Option<String>,
    length: Option<String>,
) -> Result<Option<EnclosureOut>, Unhandled> {
    let length = match length.as_deref() {
        None => Length::Absent,
        Some("") => Length::Empty,
        Some(value) => py_int(value)?.map_or(Length::Absent, Length::Int),
    };
    Ok(url
        .filter(|u| !u.is_empty())
        .map(|url| EnclosureOut { url, typ, length }))
}

/// Port of `_populate_entry_links_from_elements`. Returns the link list and
/// updates `link` the way the Python function updates `entry["link"]`.
pub fn populate_links(
    link: &mut Option<String>,
    atom_links: Vec<LinkAttrs>,
    guid_text: Option<&str>,
    guid_is_permalink: bool,
) -> Result<Vec<LinkOut>, Unhandled> {
    let mut links = Vec::with_capacity(atom_links.len());
    let mut alternate: Option<LinkOut> = None;
    for attrs in atom_links {
        let href = attrs.href.filter(|h| !h.is_empty()).or(attrs.link);
        let Some(href) = href.filter(|h| !h.is_empty()) else {
            continue;
        };
        let is_alternate = attrs.rel.as_deref() == Some("alternate");
        let skipped = matches!(attrs.rel.as_deref(), Some("edit" | "self"));
        let out = LinkOut {
            rel: attrs.rel,
            typ: attrs.typ,
            href,
            title: attrs.title,
            has_title: true,
        };
        if is_alternate && alternate.is_none() {
            alternate = Some(out);
        } else if is_alternate || !skipped {
            links.push(out);
        }
    }

    let guid_url = guid_text.filter(|g| is_http_url(g));
    if let (Some(guid), None) = (guid_url, link.as_ref()) {
        *link = Some(guid.to_owned());
        if alternate.is_some() {
            let synthetic = LinkOut {
                rel: Some("alternate".to_owned()),
                typ: Some("text/html".to_owned()),
                href: guid.to_owned(),
                title: None,
                has_title: false,
            };
            links.insert(0, synthetic);
        }
    } else if let Some(alt) = alternate {
        *link = Some(alt.href.clone());
        links.insert(0, alt);
    } else if link.is_none() && guid_is_permalink {
        // Python stores the guid text here even when it is None.
        *link = Some(
            guid_text
                .ok_or(Unhandled("permalink guid without text"))?
                .to_owned(),
        );
    }
    Ok(links)
}

/// Port of the element branch of `_populate_entry_content_preparsed`.
pub fn content_from_element(el: ContentEl) -> Result<ContentOut, Unhandled> {
    let typ = el.typ.unwrap_or_else(|| "text/html".to_owned());
    if typ == "xhtml" || typ == "application/xhtml+xml" {
        return Err(Unhandled("xhtml content"));
    }
    Ok(ContentOut {
        typ,
        language: el.language,
        base: el.base,
        value: el.text.unwrap_or_default(),
    })
}
