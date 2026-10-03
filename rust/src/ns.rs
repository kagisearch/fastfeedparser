//! The namespaces the extractor distinguishes.
use quick_xml::name::ResolveResult;

const ATOM_HTTP: &[u8] = b"http://www.w3.org/2005/Atom";
const ATOM_HTTPS: &[u8] = b"https://www.w3.org/2005/Atom";
const ATOM_03: &[u8] = b"http://purl.org/atom/ns#";
const CONTENT: &[u8] = b"http://purl.org/rss/1.0/modules/content/";
const DC: &[u8] = b"http://purl.org/dc/elements/1.1/";
const MEDIA: &[u8] = b"http://search.yahoo.com/mrss/";
pub const RDF: &[u8] = b"http://www.w3.org/1999/02/22-rdf-syntax-ns#";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ns {
    /// No namespace.
    None,
    AtomHttp,
    AtomHttps,
    Atom03,
    Content,
    Dc,
    Media,
    Other,
    /// A prefix with no declaration: not well-formed for a namespace-aware parser.
    UnknownPrefix,
}

impl Ns {
    pub fn from_resolved(resolved: &ResolveResult) -> Ns {
        match resolved {
            ResolveResult::Unbound => Ns::None,
            ResolveResult::Unknown(_) => Ns::UnknownPrefix,
            ResolveResult::Bound(ns) => match ns.as_ref() {
                ATOM_HTTP => Ns::AtomHttp,
                ATOM_HTTPS => Ns::AtomHttps,
                ATOM_03 => Ns::Atom03,
                CONTENT => Ns::Content,
                DC => Ns::Dc,
                MEDIA => Ns::Media,
                _ => Ns::Other,
            },
        }
    }

    pub fn atom_uri(self) -> Option<&'static str> {
        match self {
            Ns::AtomHttp => Some("http://www.w3.org/2005/Atom"),
            Ns::AtomHttps => Some("https://www.w3.org/2005/Atom"),
            Ns::Atom03 => Some("http://purl.org/atom/ns#"),
            _ => None,
        }
    }
}
