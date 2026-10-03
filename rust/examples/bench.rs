//! Time the native stages over a directory of feeds, without Python.
//!
//!     cargo run --release --example bench -- ../benchmark_data
use std::time::Instant;

use fastfeedparser_core::bench_api;
use quick_xml::events::Event;
use quick_xml::reader::NsReader;

fn best_ms(rounds: usize, mut run: impl FnMut() -> usize) -> (f64, usize) {
    let mut best = f64::MAX;
    let mut out = 0;
    for _ in 0..rounds {
        let start = Instant::now();
        out = run();
        best = best.min(start.elapsed().as_secs_f64() * 1000.0);
    }
    (best, out)
}

fn tokenize(data: &[u8]) -> usize {
    let mut reader = NsReader::from_reader(data);
    reader.config_mut().expand_empty_elements = true;
    let mut events = 0;
    while let Ok((_, event)) = reader.read_resolved_event() {
        if matches!(event, Event::Eof) {
            break;
        }
        events += 1;
    }
    events
}

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: bench <corpus dir> [rounds]");
    let rounds: usize = std::env::args()
        .nth(2)
        .map_or(15, |r| r.parse().expect("rounds"));
    let mut paths: Vec<_> = std::fs::read_dir(&dir)
        .expect("corpus dir")
        .map(|e| e.unwrap().path())
        .collect();
    paths.sort();
    let docs: Vec<Vec<u8>> = paths.iter().map(|p| std::fs::read(p).unwrap()).collect();
    let megabytes = docs.iter().map(Vec::len).sum::<usize>() as f64 / 1_048_576.0;
    println!("{} documents, {megabytes:.1} MB", docs.len());

    let (ms, valid) = best_ms(15, || {
        docs.iter().filter(|d| bench_api::validate(d)).count()
    });
    println!("validate (utf-8 + control chars) {ms:7.1} ms   {valid} pass");
    let (ms, events) = best_ms(15, || docs.iter().map(|d| tokenize(d)).sum());
    println!("tokenize with namespaces         {ms:7.1} ms   {events} events");
    let (ms, entries) = best_ms(rounds, || {
        docs.iter().filter_map(|d| bench_api::extract(d)).sum()
    });
    println!("extract to entries               {ms:7.1} ms   {entries} entries");
}
