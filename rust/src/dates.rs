//! Fast paths for the two date shapes that cover almost every feed.
//!
//! Each path returns a result only where it is certain to equal what the
//! Python `_parse_date` returns. Everything else is `Unknown`, and the caller
//! asks Python.
use crate::text::py_strip;

#[derive(Debug, PartialEq)]
pub enum Fast {
    Iso(String),
    /// Python returns None without trying any parser.
    NoDate,
    Unknown,
}

const MAX_DATE_CHARS: usize = 256;

pub fn fast_parse(raw: &str) -> Fast {
    if !raw.is_ascii() {
        return Fast::Unknown;
    }
    let candidate = py_strip(raw);
    if candidate.is_empty() || candidate.len() > MAX_DATE_CHARS {
        return Fast::NoDate;
    }
    let b = candidate.as_bytes();
    let starts_like_iso = b.len() >= 5 && b[4] == b'-' && b[..4].iter().all(u8::is_ascii_digit);
    if b.len() >= 20 && starts_like_iso {
        return iso(b).map_or(Fast::Unknown, Fast::Iso);
    }
    // Python rewrites or reroutes candidates like these before its RFC-822 path.
    let rerouted = b.iter().any(|&c| matches!(c, b'\n' | b'\r' | b'\t'))
        || candidate.contains("  ")
        || candidate.contains("-02-29")
        || candidate.contains("T24:")
        || candidate.contains(" 24:")
        || (b.len() >= 10 && starts_like_iso);
    if rerouted {
        return Fast::Unknown;
    }
    rfc822(b).map_or(Fast::Unknown, Fast::Iso)
}

fn digits(b: &[u8]) -> Option<u32> {
    if b.is_empty() || !b.iter().all(u8::is_ascii_digit) {
        return None;
    }
    Some(b.iter().fold(0, |n, d| n * 10 + u32::from(d - b'0')))
}

fn days_in_month(year: u32, month: u32) -> u32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ if (year.is_multiple_of(4) && !year.is_multiple_of(100)) || year.is_multiple_of(400) => {
            29
        }
        _ => 28,
    }
}

struct Civil {
    year: u32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
}

impl Civil {
    fn is_valid(&self) -> bool {
        (1..=9999).contains(&self.year)
            && (1..=12).contains(&self.month)
            && self.day >= 1
            && self.day <= days_in_month(self.year, self.month)
            && self.hour <= 23
            && self.minute <= 59
            && self.second <= 59
    }

    /// Shift a local time to UTC. None if the result leaves years 1..=9999.
    fn to_utc(&self, offset_seconds: i64) -> Option<Civil> {
        let days = days_from_civil(i64::from(self.year), self.month, self.day);
        let local =
            i64::from(self.hour) * 3600 + i64::from(self.minute) * 60 + i64::from(self.second);
        let total = days * 86400 + local - offset_seconds;
        let (year, month, day) = civil_from_days(total.div_euclid(86400));
        if !(1..=9999).contains(&year) {
            return None;
        }
        let secs = total.rem_euclid(86400) as u32;
        Some(Civil {
            year: year as u32,
            month,
            day,
            hour: secs / 3600,
            minute: secs % 3600 / 60,
            second: secs % 60,
        })
    }

    fn format(&self, microsecond: u32) -> String {
        let mut out = format!(
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        );
        if microsecond != 0 {
            out.push_str(&format!(".{microsecond:06}"));
        }
        out.push_str("+00:00");
        out
    }
}

// Howard Hinnant's civil-date algorithms (days relative to 1970-01-01).
fn days_from_civil(year: i64, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = y.div_euclid(400);
    let yoe = y.rem_euclid(400);
    let mp = i64::from((month + 9) % 12);
    let doy = (153 * mp + 2) / 5 + i64::from(day) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = yoe + era * 400 + i64::from(month <= 2);
    (year, month, day)
}

/// `YYYY-MM-DD(T| )HH:MM:SS[.fff|.ffffff](Z|z|+HH:MM|-HH:MM)`, nothing else.
fn iso(b: &[u8]) -> Option<String> {
    let punctuated = b[4] == b'-'
        && b[7] == b'-'
        && (b[10] == b'T' || b[10] == b' ')
        && b[13] == b':'
        && b[16] == b':';
    if !punctuated {
        return None;
    }
    let civil = Civil {
        year: digits(&b[0..4])?,
        month: digits(&b[5..7])?,
        day: digits(&b[8..10])?,
        hour: digits(&b[11..13])?,
        minute: digits(&b[14..16])?,
        second: digits(&b[17..19])?,
    };
    if !civil.is_valid() {
        return None;
    }
    let mut rest = &b[19..];
    let mut microsecond = 0;
    if rest.first() == Some(&b'.') {
        let len = rest[1..].iter().take_while(|c| c.is_ascii_digit()).count();
        microsecond = match len {
            3 => digits(&rest[1..4])? * 1000,
            6 => digits(&rest[1..7])?,
            _ => return None,
        };
        rest = &rest[1 + len..];
    }
    let offset_seconds = match rest {
        [b'Z'] | [b'z'] => 0,
        [sign @ (b'+' | b'-'), h1, h2, b':', m1, m2] => {
            let hours = digits(&[*h1, *h2])?;
            let minutes = digits(&[*m1, *m2])?;
            if hours > 23 || minutes > 59 {
                return None;
            }
            let seconds = i64::from(hours * 3600 + minutes * 60);
            if *sign == b'-' {
                -seconds
            } else {
                seconds
            }
        }
        _ => return None,
    };
    Some(civil.to_utc(offset_seconds)?.format(microsecond))
}

const MONTHS: [&str; 12] = [
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
];

fn named_offset(name: &[u8]) -> Option<i64> {
    Some(match name {
        b"UTC" | b"UT" | b"GMT" | b"WET" => 0,
        b"WEST" | b"BST" | b"CET" => 3600,
        b"CEST" | b"EET" => 7200,
        b"EEST" | b"MSK" => 10800,
        b"IST" => 19800,
        b"PST" | b"AKDT" => -28800,
        b"PDT" | b"MST" => -25200,
        b"MDT" | b"CST" => -21600,
        b"CDT" | b"EST" => -18000,
        b"EDT" => -14400,
        b"AKST" | b"HADT" => -32400,
        b"HST" | b"HAST" => -36000,
        b"AEST" => 36000,
        b"AEDT" => 39600,
        b"ACST" => 34200,
        b"ACDT" => 37800,
        b"AWST" | b"SGT" | b"SST" => 28800,
        b"NZST" => 43200,
        b"NZDT" => 46800,
        b"JST" | b"KST" => 32400,
        _ => return None,
    })
}

fn offset(tz: &[u8]) -> Option<i64> {
    if let [sign @ (b'+' | b'-'), rest @ ..] = tz {
        if rest.len() != 4 {
            return None;
        }
        let seconds = i64::from(digits(&rest[..2])? * 3600 + digits(&rest[2..])? * 60);
        return Some(if *sign == b'-' { -seconds } else { seconds });
    }
    if (2..=5).contains(&tz.len()) && tz.iter().all(u8::is_ascii_uppercase) {
        return named_offset(tz);
    }
    None
}

/// `[Www, ]D[D] Mon YYYY HH:MM:SS (+HHMM|-HHMM|ZONE)` with single spaces.
fn rfc822(b: &[u8]) -> Option<String> {
    let mut fields = b.split(|c| *c == b' ');
    let mut field = fields.next()?;
    if field.len() == 4 && field[3] == b',' && field[..3].iter().all(u8::is_ascii_alphabetic) {
        field = fields.next()?;
    }
    if field.len() > 2 {
        return None;
    }
    let day = digits(field)?;
    let month_name = fields.next()?;
    if month_name.len() != 3 {
        return None;
    }
    let month = MONTHS
        .iter()
        .position(|m| m.as_bytes().eq_ignore_ascii_case(month_name))? as u32
        + 1;
    let year_raw = fields.next()?;
    let time = fields.next()?;
    let tz = fields.next()?;
    if fields.next().is_some()
        || year_raw.len() != 4
        || time.len() != 8
        || time[2] != b':'
        || time[5] != b':'
    {
        return None;
    }
    let year = digits(year_raw)?;
    let (hour, minute, second) = (
        digits(&time[..2])?,
        digits(&time[3..5])?,
        digits(&time[6..])?,
    );
    let offset_seconds = offset(tz)?;
    if offset_seconds.abs() >= 86400 {
        return None;
    }
    let civil = Civil {
        year,
        month,
        day,
        hour,
        minute,
        second,
    };
    if !civil.is_valid() {
        return None;
    }
    Some(civil.to_utc(offset_seconds)?.format(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn iso_of(raw: &str) -> Option<String> {
        match fast_parse(raw) {
            Fast::Iso(s) => Some(s),
            _ => None,
        }
    }

    #[test]
    fn iso_shapes() {
        assert_eq!(
            iso_of("2024-01-15T10:30:00Z").unwrap(),
            "2024-01-15T10:30:00+00:00"
        );
        assert_eq!(
            iso_of("2024-01-15 10:30:00.123z").unwrap(),
            "2024-01-15T10:30:00.123000+00:00"
        );
        assert_eq!(
            iso_of("2024-01-15T10:30:00+02:00").unwrap(),
            "2024-01-15T08:30:00+00:00"
        );
        assert_eq!(
            iso_of("2024-01-01T00:30:00.000001+01:00").unwrap(),
            "2023-12-31T23:30:00.000001+00:00"
        );
        assert_eq!(
            iso_of("2024-02-29T23:59:59-00:30").unwrap(),
            "2024-03-01T00:29:59+00:00"
        );
    }

    #[test]
    fn rfc822_shapes() {
        assert_eq!(
            iso_of("Mon, 02 Jan 2006 15:04:05 GMT").unwrap(),
            "2006-01-02T15:04:05+00:00"
        );
        assert_eq!(
            iso_of("2 Jan 2006 15:04:05 +0000").unwrap(),
            "2006-01-02T15:04:05+00:00"
        );
        assert_eq!(
            iso_of("Mon, 02 Jan 2006 15:04:05 -0700").unwrap(),
            "2006-01-02T22:04:05+00:00"
        );
        assert_eq!(
            iso_of(" Tue, 31 Dec 2024 23:30:00 PST\n").unwrap(),
            "2025-01-01T07:30:00+00:00"
        );
    }

    #[test]
    fn undecided_input_is_left_to_python() {
        for raw in [
            "2023-02-29T10:00:00Z",
            "2024-01-15T10:30:00.1234Z",
            "2024-01-15T24:00:00Z",
            "2024-01-15T10:30:00+24:00",
            "2024-01-15T10:30:00",
            "Fri, 30 Feb 2024 10:00:00 +0100",
            // A date that cannot exist is Python's to reject, in UTC as well.
            "Wed, 45 Jan 2006 99:99:99 GMT",
            "Sat, 30 Dec 2023 23:59:60 GMT",
            "Mon, 02 Jan 2006 15:04:05 XYZ",
            "Mon,  02 Jan 2006 15:04:05 GMT",
            "Mon, 02 Jan 2006 24:04:05 GMT",
            "Mon, 02 Jan 2006 15:04:05 GMT extra",
            "yesterday",
            "0001-01-01T00:00:00+01:00",
        ] {
            assert_eq!(fast_parse(raw), Fast::Unknown, "{raw}");
        }
    }

    #[test]
    fn empty_and_overlong_are_not_dates() {
        assert_eq!(fast_parse("   "), Fast::NoDate);
        assert_eq!(fast_parse(&"1".repeat(257)), Fast::NoDate);
    }
}
