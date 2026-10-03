//! Port of `_synthesize_entry_description`: the first 512 characters of an
//! entry's content as plain text, for entries that have no description.
//!
//! Lengths and offsets are in characters, as they are in Python.
use std::borrow::Cow;

use memchr::memchr;

use crate::model::UnescapeFn;
use crate::text::{is_py_space, py_strip};

const DESCRIPTION_LEN: usize = 512;
/// Characters normalized first; the whole text only if these fall short.
const SCAN_LEN: usize = 640;
/// Tags are only stripped within this many leading characters.
const HTML_LIMIT: usize = 2048;
/// The fast path tries a prefix ending at the first ">" at or after this.
const HTML_CUT: usize = 800;

/// Byte offset of character `n`, or the length if the string is shorter.
fn char_offset(s: &str, n: usize) -> usize {
    if s.len() <= n {
        return s.len();
    }
    s.char_indices()
        .nth(n)
        .map_or(s.len(), |(offset, _)| offset)
}

fn first_chars(s: &str, n: usize) -> &str {
    &s[..char_offset(s, n)]
}

fn has_chars(s: &str, n: usize) -> bool {
    s.len() >= n && s.chars().nth(n - 1).is_some()
}

/// `re.sub(r"<[^>]+>", " ", s)`.
fn strip_tags(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let (mut copied, mut from) = (0, 0);
    while let Some(offset) = memchr(b'<', &bytes[from..]) {
        let open = from + offset;
        match memchr(b'>', &bytes[open + 1..]) {
            None => break,
            // "<>" is not a tag; the scan resumes after the "<".
            Some(0) => from = open + 1,
            Some(inner) => {
                out.push_str(&s[copied..open]);
                out.push(' ');
                copied = open + inner + 2;
                from = copied;
            }
        }
    }
    out.push_str(&s[copied..]);
    out
}

/// `" ".join(s.split())`.
fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for word in s.split(is_py_space).filter(|word| !word.is_empty()) {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(word);
    }
    out
}

fn needs_collapse(s: &str) -> bool {
    s.contains("  ") || s.bytes().any(|b| matches!(b, b'\n' | b'\t' | b'\r'))
}

fn normalize(s: &str, collapse: bool) -> String {
    if collapse {
        collapse_whitespace(s)
    } else {
        py_strip(s).to_owned()
    }
}

/// `normalize(value)[:512]` without normalizing the whole string when a
/// prefix already yields 512 characters.
fn normalized_prefix(value: &str, collapse: bool) -> String {
    if has_chars(value, SCAN_LEN + 1) {
        let head = normalize(first_chars(value, SCAN_LEN), collapse);
        if has_chars(&head, DESCRIPTION_LEN) {
            return first_chars(&head, DESCRIPTION_LEN).to_owned();
        }
    }
    first_chars(&normalize(value, collapse), DESCRIPTION_LEN).to_owned()
}

/// Port of `_html_description_from_prefix`: the description from a prefix of
/// `html`, or None when the prefix cannot be shown to give the same result
/// as the full pipeline.
fn from_prefix<E>(html: &str, unescape: &mut UnescapeFn<E>) -> Result<Option<String>, E> {
    let (low, high) = (
        char_offset(html, HTML_CUT),
        char_offset(html, HTML_LIMIT - 1),
    );
    let Some(offset) = memchr(b'>', &html.as_bytes()[low..high]) else {
        return Ok(None);
    };
    let mut head = strip_tags(&html[..=low + offset]);
    if !head.ends_with(' ') {
        return Ok(None);
    }
    if head.contains('&') {
        head = unescape(&head)?;
    }
    if !needs_collapse(&head) {
        return Ok(None);
    }
    let collapsed = collapse_whitespace(&head);
    if !has_chars(&collapsed, DESCRIPTION_LEN) {
        return Ok(None);
    }
    Ok(Some(first_chars(&collapsed, DESCRIPTION_LEN).to_owned()))
}

/// The description synthesized from `content`. `unescape` is `html.unescape`.
pub fn synthesize<E>(content: &str, unescape: &mut UnescapeFn<E>) -> Result<String, E> {
    if content.is_empty() {
        return Ok(String::new());
    }
    let mut value = Cow::Borrowed(content);
    if content.contains('<') && content.contains('>') {
        if let Some(description) = from_prefix(content, unescape)? {
            return Ok(description);
        }
        let mut stripped = strip_tags(first_chars(content, HTML_LIMIT));
        if stripped.contains('&') {
            stripped = unescape(&stripped)?;
        }
        value = Cow::Owned(stripped);
    }
    Ok(normalized_prefix(&value, needs_collapse(&value)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plain(content: &str) -> String {
        let mut identity = |s: &str| -> Result<String, ()> { Ok(s.to_owned()) };
        synthesize(content, &mut identity).unwrap()
    }

    #[test]
    fn tags_are_replaced_by_spaces() {
        assert_eq!(strip_tags("a<b>c</b>d"), "a c d");
        assert_eq!(strip_tags("a <> b <i>x</i>"), "a <> b  x ");
        assert_eq!(strip_tags("1 < 2 and <<a>b"), "1  b");
        assert_eq!(strip_tags("no close < here"), "no close < here");
    }

    #[test]
    fn whitespace_is_collapsed_like_python_split() {
        assert_eq!(collapse_whitespace(" a\u{a0}\n b\u{1f}c  "), "a b c");
    }

    #[test]
    fn short_content() {
        assert_eq!(plain(""), "");
        assert_eq!(plain("  plain text "), "plain text");
        assert_eq!(plain("<p>Hello</p>\n<p>world</p>"), "Hello world");
        assert_eq!(plain("a  b"), "a b");
    }

    #[test]
    fn long_content_is_cut_at_512_characters() {
        let text = "\u{e9}".repeat(3000);
        assert_eq!(plain(&text).chars().count(), 512);
        let html = "<p>word</p> ".repeat(500);
        let description = plain(&html);
        assert_eq!(description.chars().count(), 512);
        assert!(description.starts_with("word word word"));
    }
}
