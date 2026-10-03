//! Text decoding that matches what libxml2 hands to Python.
use memchr::{memchr, memchr3};

use crate::Unhandled;

/// `str.strip()`: Unicode White_Space plus U+001C..U+001F, which Python also
/// treats as whitespace.
pub fn py_strip(s: &str) -> &str {
    s.trim_matches(is_py_space)
}

pub fn is_py_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

fn utf8(raw: &[u8]) -> Result<&str, Unhandled> {
    simdutf8::basic::from_utf8(raw).map_err(|_| Unhandled("invalid utf-8"))
}

fn is_xml_char(c: char) -> bool {
    matches!(c, '\t' | '\n' | '\r' | '\u{20}'..='\u{d7ff}' | '\u{e000}'..='\u{fffd}' | '\u{10000}'..='\u{10ffff}')
}

/// Longest reference accepted, from "&" to ";". A longer one (a character
/// reference padded with zeros) is handed back.
const MAX_REFERENCE_LEN: usize = 16;

/// Decode the reference at the start of `b`, which begins with "&". Returns
/// the character and the reference's length.
///
/// Without a DTD only the five predefined entities and character references
/// exist. Anything else (an undefined entity, a bare "&", a reference to a
/// non-XML character) is an error for a strict parser.
fn reference(b: &[u8]) -> Result<(char, usize), Unhandled> {
    match b {
        [_, b'l', b't', b';', ..] => Ok(('<', 4)),
        [_, b'g', b't', b';', ..] => Ok(('>', 4)),
        [_, b'a', b'm', b'p', b';', ..] => Ok(('&', 5)),
        [_, b'q', b'u', b'o', b't', b';', ..] => Ok(('"', 6)),
        [_, b'a', b'p', b'o', b's', b';', ..] => Ok(('\'', 6)),
        [_, b'#', ..] => numeric_reference(b),
        _ => Err(Unhandled("entity reference")),
    }
}

fn numeric_reference(b: &[u8]) -> Result<(char, usize), Unhandled> {
    let end = b
        .iter()
        .take(MAX_REFERENCE_LEN)
        .position(|&c| c == b';')
        .ok_or(Unhandled("character reference"))?;
    let ch = match &b[2..end] {
        [b'x', hex @ ..] => char_reference(hex, 16)?,
        decimal => char_reference(decimal, 10)?,
    };
    Ok((ch, end + 1))
}

fn char_reference(digits: &[u8], radix: u32) -> Result<char, Unhandled> {
    let mut code = 0u32;
    for &d in digits {
        let value = char::from(d).to_digit(radix);
        code = value
            .and_then(|v| code.checked_mul(radix)?.checked_add(v))
            .ok_or(Unhandled("character reference"))?;
    }
    char::from_u32(code)
        .filter(|c| !digits.is_empty() && is_xml_char(*c))
        .ok_or(Unhandled("character reference"))
}

/// Append the content of a text node to `out`: "\r\n" and a lone "\r" become
/// "\n", then references are resolved (so "&#13;" stays a carriage return).
/// "]]>" is not allowed in text.
pub fn push_text(out: &mut String, raw: &[u8]) -> Result<(), Unhandled> {
    let s = utf8(raw)?;
    let bytes = s.as_bytes();
    // Decoded text is never longer than the raw text.
    out.reserve(bytes.len());
    let (mut start, mut from) = (0, 0);
    while let Some(offset) = memchr3(b'&', b'\r', b']', &bytes[from..]) {
        let at = from + offset;
        if bytes[at] == b']' {
            if bytes[at..].starts_with(b"]]>") {
                return Err(Unhandled("]]> in text"));
            }
            from = at + 1;
            continue;
        }
        out.push_str(&s[start..at]);
        if bytes[at] == b'\r' {
            out.push('\n');
            start = at + 1 + usize::from(bytes.get(at + 1) == Some(&b'\n'));
        } else {
            let (ch, len) = reference(&bytes[at..])?;
            out.push(ch);
            start = at + len;
        }
        from = start;
    }
    out.push_str(&s[start..]);
    Ok(())
}

/// Check a text node that is not being captured: its references, and that
/// it has no "]]>".
pub fn check_text(raw: &[u8]) -> Result<(), Unhandled> {
    if memchr::memmem::find(raw, b"]]>").is_some() {
        return Err(Unhandled("]]> in text"));
    }
    check_references(raw)
}

/// Check every reference in `raw`.
pub fn check_references(raw: &[u8]) -> Result<(), Unhandled> {
    let mut start = 0;
    while let Some(offset) = memchr(b'&', &raw[start..]) {
        let at = start + offset;
        start = at + reference(&raw[at..])?.1;
    }
    Ok(())
}

/// Append the content of a CDATA section to `out`, normalizing line ends.
pub fn push_cdata(out: &mut String, raw: &[u8]) -> Result<(), Unhandled> {
    let s = utf8(raw)?;
    let bytes = s.as_bytes();
    out.reserve(bytes.len());
    let mut start = 0;
    while let Some(offset) = memchr(b'\r', &bytes[start..]) {
        let at = start + offset;
        out.push_str(&s[start..at]);
        out.push('\n');
        start = at + 1 + usize::from(bytes.get(at + 1) == Some(&b'\n'));
    }
    out.push_str(&s[start..]);
    Ok(())
}

/// An attribute value: literal tabs and line breaks become spaces, and
/// references are resolved.
pub fn attr_value(raw: &[u8]) -> Result<String, Unhandled> {
    let s = utf8(raw)?;
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let (mut start, mut i) = (0, 0);
    while i < bytes.len() {
        let (replacement, len) = match bytes[i] {
            b'&' => reference(&bytes[i..])?,
            b'\r' => (' ', 1 + usize::from(bytes.get(i + 1) == Some(&b'\n'))),
            b'\t' | b'\n' => (' ', 1),
            _ => {
                i += 1;
                continue;
            }
        };
        out.push_str(&s[start..i]);
        out.push(replacement);
        i += len;
        start = i;
    }
    out.push_str(&s[start..]);
    Ok(out)
}

/// `int(value)` for the ASCII forms Python accepts. `Ok(None)` is Python's
/// ValueError. Input this function cannot decide exactly is handed back.
pub fn py_int(value: &str) -> Result<Option<i64>, Unhandled> {
    if !value.is_ascii() {
        return Err(Unhandled("non-ascii integer"));
    }
    let s = py_strip(value);
    let (negative, digits) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    let malformed = digits.is_empty()
        || digits.starts_with('_')
        || digits.ends_with('_')
        || digits.contains("__")
        || !digits.bytes().all(|b| b.is_ascii_digit() || b == b'_');
    if malformed {
        return Ok(None);
    }
    let plain: String = digits.chars().filter(|c| *c != '_').collect();
    match plain.parse::<i64>() {
        Ok(n) => Ok(Some(if negative { -n } else { n })),
        Err(_) => Err(Unhandled("integer out of range")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_matches_python_whitespace_set() {
        // Every code point for which Python's str.isspace() is true.
        let py_spaces = [
            0x9, 0xa, 0xb, 0xc, 0xd, 0x1c, 0x1d, 0x1e, 0x1f, 0x20, 0x85, 0xa0, 0x1680, 0x2000,
            0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200a, 0x2028,
            0x2029, 0x202f, 0x205f, 0x3000,
        ];
        for cp in 0..=0x10ffffu32 {
            if let Some(c) = char::from_u32(cp) {
                assert_eq!(is_py_space(c), py_spaces.contains(&cp), "U+{cp:04X}");
            }
        }
    }

    #[test]
    fn eol_is_normalized_before_references() {
        let mut out = String::new();
        push_text(&mut out, b"a\r\nb\rc&#13;d&amp;e").unwrap();
        assert_eq!(out, "a\nb\nc\rd&e");
    }

    #[test]
    fn undefined_entity_and_bad_reference_are_handed_back() {
        let mut out = String::new();
        assert!(push_text(&mut out, b"a&nbsp;b").is_err());
        assert!(push_text(&mut out, b"AT&T").is_err());
        assert!(push_text(&mut out, b"a&#31;b").is_err());
        assert!(push_text(&mut out, b"a ]]> b").is_err());
        assert!(check_text(b"a&nbsp;b").is_err());
        assert!(check_text(b"a ]]> b").is_err());
        assert!(check_text(b"plain ]] > &amp;").is_ok());
        assert!(check_references(b"]]> is fine in an attribute &lt;").is_ok());
    }

    #[test]
    fn attribute_whitespace_becomes_spaces() {
        assert_eq!(attr_value(b"a\tb\r\nc&#10;d").unwrap(), "a b c\nd");
        assert_eq!(attr_value(b"x&lt;y\rz").unwrap(), "x<y z");
    }

    #[test]
    fn references_are_decoded_strictly() {
        let mut out = String::new();
        push_text(&mut out, b"&lt;&gt;&amp;&quot;&apos;&#65;&#x41;&#x1F600;").unwrap();
        assert_eq!(out, "<>&\"'AA\u{1F600}");
        for bad in [
            &b"&LT;"[..],
            b"&#;",
            b"&#x;",
            b"&#X41;",
            b"&#0;",
            b"&#xD800;",
            b"&#xFFFE;",
            b"&;",
            b"&amp",
            b"&#00000000000000065;",
        ] {
            assert!(
                push_text(&mut String::new(), bad).is_err(),
                "{:?}",
                std::str::from_utf8(bad)
            );
            assert!(check_text(bad).is_err());
        }
    }

    #[test]
    fn cdata_keeps_references_and_normalizes_line_ends() {
        let mut out = String::new();
        push_cdata(&mut out, b"a&amp;b\r\nc\r").unwrap();
        assert_eq!(out, "a&amp;b\nc\n");
    }

    #[test]
    fn int_follows_python() {
        assert_eq!(py_int(" 10 ").unwrap(), Some(10));
        assert_eq!(py_int("-7").unwrap(), Some(-7));
        assert_eq!(py_int("1_000").unwrap(), Some(1000));
        assert_eq!(py_int("010").unwrap(), Some(10));
        assert_eq!(py_int("1e3").unwrap(), None);
        assert_eq!(py_int("").unwrap(), None);
        assert_eq!(py_int("1__0").unwrap(), None);
        assert_eq!(py_int("10.0").unwrap(), None);
        assert!(py_int("99999999999999999999").is_err());
    }
}
