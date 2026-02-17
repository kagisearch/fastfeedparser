import argparse
import time

import fastfeedparser
import feedparser
import httpx

# Test feeds
feeds = [
    "https://feedpress.me/FIJ",
    "https://techtinkering.com/feed.xml",
    "https://glineq.blogspot.com/feeds/posts/default",
    "https://stml.tumblr.com/rss",
    "http://feeds.feedburner.com/mishadoff",
    "https://www.speakingbody.com/rss/",
    "https://emacsninja.com/feed.atom",
    "http://causality.cs.ucla.edu/blog/index.php/feed/",
    "https://blog.railsapps.org/rss",
    "https://www.bravelysheblogs.com/feed/",
    "https://realphysics.blogspot.com/feeds/posts/default",
    "https://staysaasy.com/feed.xml",
    "https://explog.in/rss.xml",
    "https://planet.clojure.in/atom.xml",
    "https://www.youtube.com/feeds/videos.xml?channel_id=UCsE74YJvPJpaquzTPMO8hAA",
    "https://www.petekeen.net/index.xml",
    "https://jelleraaijmakers.nl/feed",
    "https://fale.io/index.xml",
    "https://gessfred.xyz/rss.xml",
    "https://fanf.dreamwidth.org/data/rss",
    "https://bernsteinbear.com/feed.xml",
    "https://feeds.kottke.org/main",
    "https://dkg.fifthhorseman.net/blog/feeds/all.atom.xml",
    "https://zhubert.com/index.xml",
    "https://lovergne.dev/rss.xml",
    "https://blog.kagi.com/rss.xml",
    "https://battlefieldanomalies.com/home/feed/",
    "http://davidbau.com/index.rdf",
    "https://blog.kagamino.dev/index.xml",
    "https://feeds.transistor.fm/fallthrough",
    "http://dontcodetired.com/blog/syndication.axd",
    "https://aivarsk.com/atom.xml",
    "http://markcoddington.com/feed/",
    "https://bendauphinee.com/writing/feed/",
    "https://www.oscardom.dev/index.xml",
    "https://alwaystwisted.com/feed.xml",
    "https://killjoy.bearblog.dev/rss.xml",
    "https://nithinbekal.com/feed.xml",
    "https://blog.emacsen.net/atom.xml",
    "https://therecouldhavebeensnakes.wordpress.com/feed/",
    "https://journal.jatan.space/rss/",
    "https://alexwlchan.net/atom.xml",
    "https://telescoper.blog/feed/",
    "https://blog.knatten.org/feed/",
    "https://timtech.blog/feed/feed.xml",
    "https://iampaulbrown.com/rss",
    "https://benlog.com/feed/",
    "https://raggywaltz.com/feed/",
    "https://herman.bearblog.dev/feed/",
    "https://dylanharris.org/feed-me.rss",
    "https://eliot-jones.com/rss",
    "https://blog.kroy.io/feed/",
    "https://jfg-mysql.blogspot.com/feeds/posts/default",
    "https://dzidas.com/atom.xml",
    "https://ariannasimpson.com/blog/feed/",
    "https://www.everydayislikewednesday.com/atom.xml",
    "https://www.bastibl.net/atom.xml",
    "https://yuxi.ml/feeds.xml",
    "https://bugramming.dev/index.xml",
    "https://evanfields.github.io/feed.xml",
    "https://raahel.bearblog.dev/atom/",
    "https://mahdytech.com/rss.xml",
    "https://fogblog-hermansheephouse.blogspot.com/feeds/posts/default",
    "https://ctoomey.com/atom.xml",
    "https://blog.lasheen.dev/index.xml",
    "https://markheath.net/feed/rss",
    "https://stancarney.co/rss/",
    "https://thecretefleet.com/blog/f.atom",
    "https://anteru.net/rss.xml",
    "https://blog.drewolson.org/index.xml",
    "https://blog.noredink.com/rss",
    "https://glasspetalsmoke.blogspot.com/feeds/posts/default",
    "https://feeds.washingtonpost.com/rss/world",
    "https://abcnews.go.com/abcnews/internationalheadlines",
    "https://aljazeera.com/xml/rss/all.xml",
    "https://allafrica.com/tools/headlines/rdf/latest/headlines.rdf",
    "https://remimercier.com/feed.xml",
    "https://en.mercopress.com/rss/",
    "https://feeds.a.dj.com/rss/RSSWorldNews.xml",
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://feeds.elpais.com/mrss-s/pages/ep/site/english.elpais.com/portada",
    "https://feeds.feedburner.com/ndtvnews-world-news",
    "https://feeds.npr.org/1004/rss.xml",
    "https://foreignpolicy.com/feed/",
    "https://japantoday.com/category/world/feed",
    "https://restofworld.org/feed/latest",
    "https://rss.csmonitor.com/feeds/all",
    "https://rss.dw.com/rdf/rss-en-all",
    "https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
    "https://theweek.com/feeds.xml",
    "https://time.com/world/feed",
    "https://www.abc.net.au/news/feed/45910/rss.xml",
    "https://www.al-monitor.com/rss",
    "https://www.boston.com/tag/world-news/feed/",
    "https://www.cbsnews.com/latest/rss/world",
    "https://www.dawn.com/feeds/world/",
    "https://www.economist.com/asia/rss.xml",
    "https://time.com/tech/feed/",
    "http://stratechery.com/feed/",
    "https://www.404media.co/rss/",
]


headers = {
    "User-Agent": "Mozilla/5.0 fastfeedparser",
    "Accept": "*/*",
    "Connection": "close",
}


def process_feed(url, skip_feedparser=False, iterations=3):
    """Process a single feed and return timing results."""
    result = {
        "url": url,
        "ffp_time": 0,
        "fp_time": 0,
        "ffp_entries": 0,
        "fp_entries": 0,
        "success": False,
    }

    try:
        # Create client per request for ProcessPoolExecutor compatibility
        with httpx.Client(verify=False) as client:
            resp = client.get(url, timeout=20.0, follow_redirects=True, headers=headers)
            content = resp.content

            # Test fastfeedparser
            try:
                start_time = time.perf_counter()
                for _ in range(iterations):
                    feed = fastfeedparser.parse(content)
                result["ffp_time"] = (time.perf_counter() - start_time) / iterations
                result["ffp_entries"] = len(feed.entries)
                print(
                    f"[{url}] FastFeedParser: {len(feed.entries)} entries in {result['ffp_time']:.3f}s (avg of {iterations} runs)"
                )
            except Exception as e:
                result["ffp_time"] = (time.perf_counter() - start_time) / iterations
                print(f"[{url}] FastFeedParser failed: {e}")

            # Test feedparser
            if not skip_feedparser:
                try:
                    start_time = time.perf_counter()
                    for _ in range(iterations):
                        feed = feedparser.parse(content)
                    result["fp_time"] = (time.perf_counter() - start_time) / iterations
                    result["fp_entries"] = len(feed.entries)
                    print(
                        f"[{url}] Feedparser: {len(feed.entries)} entries in {result['fp_time']:.3f}s (avg of {iterations} runs)"
                    )
                except Exception as e:
                    result["fp_time"] = (time.perf_counter() - start_time) / iterations
                    print(f"[{url}] Feedparser failed: {e}")

            if skip_feedparser:
                if result["ffp_time"] > 0:
                    result["success"] = True
            else:
                if result["ffp_time"] > 0 and result["fp_time"] > 0:
                    result["success"] = True
                    print(f"[{url}] Speedup: {result['fp_time']/result['ffp_time']:.1f}x")

    except Exception as e:
        print(f"[{url}] Failed to fetch feed: {e}")

    return result


def test_parsers(skip_feedparser=False, iterations=3):
    print("Testing feed parsers...")
    if skip_feedparser:
        print("Running in FastFeedParser-only mode (-s)")
    print(f"Processing feeds sequentially (no parallelization)...")
    print(f"Each feed will be parsed {iterations} times for accurate timing")
    print("-" * 50)

    results = []
    overall_start_time = time.perf_counter()

    # Simple sequential loop - no parallelization
    for url in feeds:
        try:
            result = process_feed(url, skip_feedparser, iterations)
            results.append(result)
        except Exception as e:
            print(f"Exception processing {url}: {e}")

    overall_time = time.perf_counter() - overall_start_time

    # Calculate totals
    total_ffp_time = sum(r["ffp_time"] for r in results)
    total_fp_time = sum(r["fp_time"] for r in results)
    total_ffp_entries = sum(r["ffp_entries"] for r in results)
    total_fp_entries = sum(r["fp_entries"] for r in results)
    successful_feeds = sum(1 for r in results if r["success"])

    # Find outliers
    entry_mismatches = []
    slow_feeds = []

    for r in results:
        if not r["success"]:
            continue

        # Check for entry count mismatches
        if not skip_feedparser and r["ffp_entries"] != r["fp_entries"]:
            entry_mismatches.append({
                "url": r["url"],
                "ffp_entries": r["ffp_entries"],
                "fp_entries": r["fp_entries"],
                "diff": r["ffp_entries"] - r["fp_entries"]
            })

        # Check for slow performance (less than 1.1x speedup)
        if not skip_feedparser and r["fp_time"] > 0 and r["ffp_time"] > 0:
            speedup = r["fp_time"] / r["ffp_time"]
            if speedup < 1.1:
                slow_feeds.append({
                    "url": r["url"],
                    "speedup": speedup,
                    "ffp_time": r["ffp_time"],
                    "fp_time": r["fp_time"]
                })

    print("\nSummary:")
    print("-" * 50)
    print(f"Total wall-clock time: {overall_time:.2f}s (with parallel execution)")
    print(f"Successfully tested {successful_feeds}/{len(feeds)} feeds")
    if successful_feeds > 0:
        print(f"\nFastFeedParser:")
        print(f"  Total entries: {total_ffp_entries}")
        print(f"  Total parsing time: {total_ffp_time:.2f}s")
        print(f"  Average per feed: {total_ffp_time/successful_feeds:.3f}s")

        if not skip_feedparser:
            print(f"\nFeedparser:")
            print(f"  Total entries: {total_fp_entries}")
            print(f"  Total parsing time: {total_fp_time:.2f}s")
            print(f"  Average per feed: {total_fp_time/successful_feeds:.3f}s")
            print(
                f"\nSpeedup: FastFeedParser is {(total_fp_time/total_ffp_time):.1f}x faster"
            )

            # Report outliers
            if entry_mismatches:
                print(f"\n⚠️  OUTLIERS: Entry Count Mismatches ({len(entry_mismatches)} feeds)")
                print("-" * 50)
                for m in entry_mismatches:
                    print(f"  {m['url']}")
                    print(f"    FastFeedParser: {m['ffp_entries']} entries")
                    print(f"    Feedparser: {m['fp_entries']} entries")
                    print(f"    Difference: {m['diff']:+d}")

            if slow_feeds:
                print(f"\n⚠️  OUTLIERS: Slow Performance (<1.1x speedup, {len(slow_feeds)} feeds)")
                print("-" * 50)
                slow_feeds.sort(key=lambda x: x["speedup"])
                for s in slow_feeds:
                    print(f"  {s['url']}")
                    print(f"    Speedup: {s['speedup']:.2f}x")
                    print(f"    FastFeedParser: {s['ffp_time']*1000:.2f}ms")
                    print(f"    Feedparser: {s['fp_time']*1000:.2f}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark feed parsers")
    parser.add_argument(
        "-s",
        "--skip-feedparser",
        action="store_true",
        help="Skip feedparser and run only fastfeedparser",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=3,
        help="Number of iterations to run for each feed (default: 3)",
    )
    args = parser.parse_args()

    test_parsers(
        skip_feedparser=args.skip_feedparser,
        iterations=args.iterations,
    )
