//! Checks for input a strict XML parser rejects but the tokenizer accepts.
//! Such documents are handed back so the lxml path deals with them.
use memchr::memmem;
use quick_xml::events::attributes::Attribute;
use quick_xml::events::{BytesDecl, BytesStart};
use quick_xml::name::ResolveResult;
use quick_xml::reader::NsReader;

use crate::model::Unhandled;
use crate::text::{attr_value, check_text};

pub fn document(data: &[u8]) -> Result<(), Unhandled> {
    simdutf8::basic::from_utf8(data).map_err(|_| Unhandled("invalid utf-8"))?;
    // Branch-free per chunk so the compiler can vectorize the scan.
    let has_control = data.chunks(4096).any(|chunk| {
        chunk.iter().fold(false, |found, &b| {
            found | ((b < 0x20) & (b != b'\t') & (b != b'\n') & (b != b'\r'))
        })
    });
    if has_control {
        return Err(Unhandled("control character"));
    }
    Ok(())
}

/// The XML declaration: version 1.0, then optionally a UTF-8 compatible
/// encoding, then optionally standalone, in that order and nothing else.
pub fn declaration(decl: &BytesDecl) -> Result<(), Unhandled> {
    let bad = Unhandled("xml declaration");
    let raw = decl
        .strip_prefix(b"xml")
        .ok_or(Unhandled("xml declaration"))?;
    let mut expected = [&b"version"[..], b"encoding", b"standalone"].into_iter();
    for_each_attribute(raw, |name, value| {
        let valid = match expected.by_ref().find(|n| *n == name) {
            Some(b"version") => value == b"1.0",
            Some(b"encoding") => {
                value.eq_ignore_ascii_case(b"utf-8") || value.eq_ignore_ascii_case(b"us-ascii")
            }
            Some(_) => value == b"yes" || value == b"no",
            None => false,
        };
        if valid {
            Ok(())
        } else {
            Err(Unhandled("xml declaration"))
        }
    })?;
    // "version" is consumed from `expected` only when it was present first.
    if raw.trim_ascii_start().starts_with(b"version") {
        Ok(())
    } else {
        Err(bad)
    }
}

fn is_name_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_' || b >= 0x80
}

fn is_ncname(part: &[u8]) -> bool {
    part.first().is_some_and(|&b| is_name_start(b))
        && part
            .iter()
            .all(|&b| is_name_start(b) || b.is_ascii_digit() || b == b'.' || b == b'-')
}

/// An element or attribute name: `local` or `prefix:local`. Non-ASCII bytes
/// are accepted without checking their Unicode class.
pub fn name(name: &[u8]) -> Result<(), Unhandled> {
    let mut parts = name.splitn(2, |&b| b == b':');
    let valid = parts.all(is_ncname);
    if valid {
        Ok(())
    } else {
        Err(Unhandled("element or attribute name"))
    }
}

/// Strict start-tag syntax, which the tokenizer does not enforce: every
/// attribute is preceded by whitespace and has a quoted value. `raw` is the
/// start tag after the element name.
pub fn attributes_syntax(raw: &[u8]) -> Result<(), Unhandled> {
    for_each_attribute(raw, |_, _| Ok(()))
}

/// Walk `name="value"` pairs under the strict syntax, calling `visit` with
/// the raw name and value of each.
fn for_each_attribute(
    raw: &[u8],
    mut visit: impl FnMut(&[u8], &[u8]) -> Result<(), Unhandled>,
) -> Result<(), Unhandled> {
    let bad = Unhandled("attribute syntax");
    let is_space = |b: &u8| matches!(b, b' ' | b'\t' | b'\r' | b'\n');
    let skip_space = |from: usize| from + raw[from..].iter().take_while(|b| is_space(b)).count();
    let mut i = 0;
    loop {
        let name_start = skip_space(i);
        if name_start == raw.len() {
            return Ok(());
        }
        if name_start == i {
            return Err(bad);
        }
        let name_len = raw[name_start..]
            .iter()
            .take_while(|b| !is_space(b) && **b != b'=')
            .count();
        let equals = skip_space(name_start + name_len);
        if name_len == 0 || raw.get(equals) != Some(&b'=') {
            return Err(bad);
        }
        let quote_at = skip_space(equals + 1);
        let Some(quote @ (b'"' | b'\'')) = raw.get(quote_at) else {
            return Err(bad);
        };
        let value_start = quote_at + 1;
        let Some(value_len) = memchr::memchr(*quote, &raw[value_start..]) else {
            return Err(bad);
        };
        visit(
            &raw[name_start..name_start + name_len],
            &raw[value_start..value_start + value_len],
        )?;
        i = value_start + value_len + 1;
    }
}

/// A processing instruction: a valid target, then whitespace or the end.
/// The target "xml" in any letter case is reserved.
pub fn processing_instruction(raw: &[u8]) -> Result<(), Unhandled> {
    let target_len = raw.iter().take_while(|b| !b.is_ascii_whitespace()).count();
    let target = &raw[..target_len];
    if is_ncname(target) && !target.eq_ignore_ascii_case(b"xml") {
        Ok(())
    } else {
        Err(Unhandled("processing instruction"))
    }
}

/// "]]>" may not appear in character data.
pub fn text(raw: &[u8]) -> Result<(), Unhandled> {
    match memmem::find(raw, b"]]>") {
        Some(_) => Err(Unhandled("]]> in text")),
        None => Ok(()),
    }
}

/// "--" may not appear inside a comment, and it may not end with "-".
pub fn comment(raw: &[u8]) -> Result<(), Unhandled> {
    if memmem::find(raw, b"--").is_some() || raw.last() == Some(&b'-') {
        return Err(Unhandled("comment syntax"));
    }
    Ok(())
}

/// Checks for an attribute whose value is not otherwise read.
fn attribute(reader: &NsReader<&[u8]>, attr: &Attribute) -> Result<(), Unhandled> {
    check_text(&attr.value)?;
    attribute_name(reader, attr)
}

/// The name must be well-formed and use a declared prefix; a namespace
/// declaration may not bind a prefix to the empty string.
pub fn attribute_name(reader: &NsReader<&[u8]>, attr: &Attribute) -> Result<(), Unhandled> {
    let key = attr.key.as_ref();
    name(key)?;
    if memchr::memchr(b'<', &attr.value).is_some() {
        return Err(Unhandled("< in attribute value"));
    }
    if key.starts_with(b"xmlns:") {
        return if attr.value.is_empty() {
            Err(Unhandled("empty namespace binding"))
        } else {
            Ok(())
        };
    }
    if !key.contains(&b':') || key.starts_with(b"xml:") {
        return Ok(());
    }
    match reader.resolve_attribute(attr.key).0 {
        ResolveResult::Unknown(_) => Err(Unhandled("undeclared attribute prefix")),
        _ => Ok(()),
    }
}

/// Read the named attributes of `e`, checking every attribute on the way.
pub fn read_attrs<const N: usize>(
    reader: &NsReader<&[u8]>,
    e: &BytesStart,
    names: [&[u8]; N],
) -> Result<[Option<String>; N], Unhandled> {
    const NONE: Option<String> = None;
    let mut out = [NONE; N];
    for attr in e.attributes() {
        let attr = attr.map_err(|_| Unhandled("attribute syntax"))?;
        match names.iter().position(|name| *name == attr.key.as_ref()) {
            Some(i) => {
                attribute_name(reader, &attr)?;
                out[i] = Some(attr_value(&attr.value)?);
            }
            None => attribute(reader, &attr)?,
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        for good in [
            &b"a"[..],
            b"_a",
            b"a1",
            b"a.b-c",
            b"media:content",
            b"\xc3\xa9t\xc3\xa9",
        ] {
            assert!(name(good).is_ok(), "{good:?}");
        }
        for bad in [
            &b""[..],
            b"1a",
            b"&rl",
            b"a:b:c",
            b":a",
            b"a:",
            b"a b",
            b"a:1",
            b"-a",
        ] {
            assert!(name(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn attribute_syntax() {
        for good in [
            &b""[..],
            b" ",
            b" a=\"1\"",
            b" a = '1'  b=\"2\"\n",
            b" a=\"x='y'\"",
        ] {
            assert!(
                attributes_syntax(good).is_ok(),
                "{:?}",
                std::str::from_utf8(good)
            );
        }
        for bad in [
            &b" a=\"1\"b=\"2\""[..],
            b" a=1",
            b" a",
            b" =\"1\"",
            b" a=\"1",
            b"a=\"1\"",
        ] {
            assert!(
                attributes_syntax(bad).is_err(),
                "{:?}",
                std::str::from_utf8(bad)
            );
        }
    }

    #[test]
    fn declarations() {
        let check = |content: &str| {
            let xml = format!("<?{content}?><a/>");
            let mut reader = quick_xml::Reader::from_str(&xml);
            match reader.read_event().unwrap() {
                quick_xml::events::Event::Decl(decl) => declaration(&decl).is_ok(),
                other => panic!("not a declaration: {other:?}"),
            }
        };
        for good in [
            "xml version=\"1.0\"",
            "xml version='1.0' encoding='UTF-8'",
            "xml version=\"1.0\" encoding=\"utf-8\" standalone=\"yes\" ",
            "xml version=\"1.0\" standalone='no'",
        ] {
            assert!(check(good), "{good}");
        }
        for bad in [
            "xml version=\"1.1\"",
            "xml encoding=\"utf-8\"",
            "xml version=\"1.0\"encoding=\"utf-8\"",
            "xml version=\"1.0\" standalone=\"maybe\"",
            "xml version=\"1.0\" encoding=\"latin-1\"",
            "xml version=\"1.0\" standalone=\"yes\" encoding=\"utf-8\"",
            "xml version=\"1.0\" foo=\"1\"",
            "xml version=1.0",
            "xml",
            "xml version=\"1.0\" version=\"1.0\"",
        ] {
            assert!(!check(bad), "{bad}");
        }
    }

    #[test]
    fn processing_instructions() {
        for good in [&b"pi"[..], b"pi data", b"xml-stylesheet href='a'"] {
            assert!(
                processing_instruction(good).is_ok(),
                "{:?}",
                std::str::from_utf8(good)
            );
        }
        for bad in [
            &b"xmlversion=\"1.0\""[..],
            b"xml&e; version='1.0'",
            b"xml' version",
            b"XML version='1.0'",
            b"",
            b" pi",
            b"1pi",
        ] {
            assert!(
                processing_instruction(bad).is_err(),
                "{:?}",
                std::str::from_utf8(bad)
            );
        }
    }

    #[test]
    fn text_and_comment_rules() {
        assert!(text(b"a ]] > b").is_ok());
        assert!(text(b"a ]]> b").is_err());
        assert!(comment(b" fine - really ").is_ok());
        assert!(comment(b" a -- b ").is_err());
        assert!(comment(b" a -").is_err());
    }
}
