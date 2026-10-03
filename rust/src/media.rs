//! Port of `_parse_media_content`.
use crate::common::stripped;
use crate::model::{MediaOut, Text};

/// First media:description / media:credit child of an element, if any. The
/// inner Option is the element's text.
pub type FirstDesc = Option<Option<Text>>;
pub type FirstCredit = Option<(Option<Text>, Option<String>)>;

#[derive(Default)]
struct MediaItem {
    out: MediaOut,
    has_description_el: bool,
    has_credit_el: bool,
}

pub struct Thumb {
    pub url: Option<String>,
    pub width: Option<i64>,
    pub height: Option<i64>,
}

#[derive(Default)]
pub struct MediaAcc {
    items: Vec<MediaItem>,
    /// media:thumbnail elements whose parent is not a media:content.
    thumbs: Vec<Thumb>,
}

impl MediaAcc {
    /// Register a media:content element; returns its index.
    pub fn begin(&mut self, out: MediaOut) -> usize {
        self.items.push(MediaItem {
            out,
            ..MediaItem::default()
        });
        self.items.len() - 1
    }

    pub fn add_thumb(&mut self, thumb: Thumb) {
        self.thumbs.push(thumb);
    }

    pub fn set_title(&mut self, idx: usize, title: &Option<Option<Text>>) {
        if let Some(text) = title {
            self.items[idx].out.title = stripped(text);
        }
    }

    pub fn set_text(&mut self, idx: usize, text: &Option<Option<Text>>) {
        if let Some(text) = text {
            self.items[idx].out.text = stripped(text);
        }
    }

    pub fn set_thumbnail(&mut self, idx: usize, url: Option<Option<String>>) {
        if let Some(url) = url {
            self.items[idx].out.thumbnail_url = url;
        }
    }

    /// Use `desc` and `credit` for whichever of the two the item still lacks.
    /// Called first with the item's own children, then with its parent's.
    pub fn fill(&mut self, idx: usize, desc: &FirstDesc, credit: &FirstCredit) {
        let item = &mut self.items[idx];
        if let (false, Some(text)) = (item.has_description_el, desc) {
            item.has_description_el = true;
            item.out.description = stripped(text);
        }
        if let (false, Some((text, scheme))) = (item.has_credit_el, credit) {
            item.has_credit_el = true;
            if text.is_some() {
                item.out.credit = stripped(text);
                item.out.credit_scheme = scheme.clone();
            }
        }
    }

    pub fn needs_parent(&self, idx: usize) -> bool {
        !self.items[idx].has_description_el || !self.items[idx].has_credit_el
    }

    pub fn finish(self) -> Vec<MediaOut> {
        let contents: Vec<MediaOut> = self
            .items
            .into_iter()
            .map(|i| i.out)
            .filter(|o| !is_empty(o))
            .collect();
        if !contents.is_empty() {
            return contents;
        }
        self.thumbs
            .into_iter()
            .map(|t| MediaOut {
                url: t.url,
                typ: Some("image/jpeg".to_owned()),
                width: t.width,
                height: t.height,
                ..MediaOut::default()
            })
            .collect()
    }
}

fn is_empty(o: &MediaOut) -> bool {
    o.url.is_none()
        && o.typ.is_none()
        && o.medium.is_none()
        && o.width.is_none()
        && o.height.is_none()
        && o.title.is_none()
        && o.text.is_none()
        && o.description.is_none()
        && o.credit.is_none()
        && o.credit_scheme.is_none()
        && o.thumbnail_url.is_none()
}
