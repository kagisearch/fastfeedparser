import argparse
import hashlib
import os
import statistics
import time

import fastfeedparser
import feedparser
import httpx

BENCHMARK_DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "benchmark_data"
)

FEEDS = [
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
    # Kagi Small Web OPML picks
    "https://www.makeartwithpython.com/blog/feed.xml",
    "https://myhsu.xyz/index.xml",
    "https://tomhazledine.com/feed.xml",
    "https://www.chesterton.org/feed/",
    "https://medwa.io/feed.xml",
    "https://juliawise.net/feed/",
    "https://adam.scherlis.com/feed/",
    "https://raymondyoo.com/rss/",
    "https://technobabble.bearblog.dev/rss.xml",
    "https://deepankarm.github.io/index.xml",
    "https://raywoodcockslatest.wordpress.com/feed/",
    "https://tkhan.blog/atom/",
    "http://james-simon.github.io/feed.xml",
    "https://boat.karlnelson.net/index.xml",
    "https://lunar-phase.the-comic.org/rss/",
    "https://collectiveaction.tech/feed",
    "https://willpatrick.xyz/feed.xml",
    "https://elijahpotter.dev/rss.xml",
    "https://www.thatsoftwaredude.com/rss",
    "https://logits.bearblog.dev/atom",
    "https://learnetto.com/blog/rss",
    "https://coghlan.me/feed.xml",
    "https://areweanticheatyet.com/feed.rss",
    "https://www.usabilitycounts.com/feed/",
    "https://www.math.columbia.edu/~woit/wordpress/?feed=rss2",
    "https://oceanbites.org/feed/atom/",
    "https://mikhail.io/feed/",
    "https://avandeursen.com/feed/",
    "https://zerokspot.com/index.xml",
    "https://www.alexblackie.com/index.xml",
    "https://keymaterial.net/feed/",
    "https://www.youtube.com/feeds/videos.xml?channel_id=UC4Otk-uDioJN0tg6s1QO9lw",
    "https://www.morningtempo.com/feed/",
    "https://andydote.co.uk/rss.xml",
    "https://higherorderlogic.com/feed.xml",
    "https://ninaklose.de/feed/",
    "https://hetmehta.com/rss.xml",
    "https://daridor.blog/atom/",
    "https://arsoncafe.blogspot.com/feeds/posts/default",
    "https://lordofnothing.com/atom/",
    "https://kittenlabs.de/index.xml",
    "https://spoon-tamago.com/feed/",
    "https://racc.blog/atom/",
    "https://www.mrspeaker.net/feed/",
    "https://lookgreatnaked.com/atom/",
    "https://www.dylanpaulus.com/rss.xml",
    "https://jessimekirk.com/feed.xml",
    "https://thomashoneyman.com/index.xml",
    "https://f1metrics.wordpress.com/feed/",
    "https://pooriat.com/index.xml",
    "https://blog.thecaptain.dev/atom/",
    "https://fulgidus.github.io/rss.xml",
    "https://www.youtube.com/feeds/videos.xml?channel_id=UCotwjyJnb-4KW7bmsOoLfkg",
    "https://www.richardrodger.com/feed/",
    "https://www.peterme.com/feed/",
    "https://jtemporal.com/feed.xml",
    "https://geohot.github.io/blog/feed.xml",
    "https://serversforhackers.com/feed/",
    "https://pikku.dev/blog/atom.xml",
    "https://www.workingsoftware.dev/rss/",
    "https://blog.poisson.chat/rss.xml",
    "https://www.massicotte.org/feed.xml",
    "https://langorigami.com/atom/",
    "https://blog.gaurang.page/rss.xml",
    "https://blog.dmcc.io//index.xml",
    "https://tomlibertiny.com/feed/",
    "https://feeds.feedburner.com/SalmonRun",
    "https://shep.ca/feed/",
    "https://www.loudandquiet.com/feed/atom/",
    "https://blog.hboeck.de/rss.php",
    "https://rants.org/feed/",
    "https://chaidarun.com/feed.xml",
    "https://www.wellappointeddesk.com/feed/",
    "https://journal.miso.town/atom?url=https://wiki.xxiivv.com/site/now.html",
    "https://recaffeinate.co/index.xml",
    "https://densumesh.dev/rss.xml",
    "https://www.youtube.com/feeds/videos.xml?channel_id=UCHS55yDvORmCpbM_3vNQQsQ",
    "https://www.oldhousedreams.com/feed/atom/",
    "http://www.warrenburt.com/journal/atom.xml",
    "https://sencjw.com/atom.xml",
    "https://davidduchemin.com/feed/",
    "https://markgreville.ie/feed/",
    "https://codearcana.com/feeds/all.atom.xml",
    "https://minutestomidnight.co.uk/feed.xml",
    "https://simon-davies.name/posts/feed.rss",
    "https://www.pagetable.com/?feed=rss2",
    "https://amitgawande.com/feed",
    "https://humanwhocodes.com/feeds/all.json",
    "https://mridul.io/feed.xml",
    "https://kopiascsaba.hu/blog/index.xml",
    "https://pavursec.com/index.xml",
    "https://discourse.ardour.org/c/blog/15.rss",
    "https://vince-debian.blogspot.com/feeds/posts/default",
    "https://albanbrooke.com/feed/",
    "https://thundergolfer.com/feed.xml",
    "https://tarantsov.com/rss.xml",
    "https://xaviergeerinck.com/rss/",
    "https://afarshadblog.bearblog.dev/atom.xml",
    "https://frisk.space/index.xml",
    "https://www.youtube.com/feeds/videos.xml?channel_id=UCA7LqFpF_Zlty4Yg2iGJppw",
    "https://quartzlibrary.com/atom/",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 fastfeedparser",
    "Accept": "*/*",
    "Connection": "close",
}


def _cache_path(url):
    """Return the file path for a cached feed URL."""
    name = hashlib.sha256(url.encode()).hexdigest()[:16]
    return os.path.join(BENCHMARK_DATA_DIR, name)


def _load_content(url):
    """Load feed content from cache, or fetch live if not cached."""
    path = _cache_path(url)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    with httpx.Client(verify=False) as client:
        resp = client.get(url, timeout=20.0, follow_redirects=True, headers=HEADERS)
        os.makedirs(BENCHMARK_DATA_DIR, exist_ok=True)
        with open(path, "wb") as f:
            f.write(resp.content)
        return resp.content


def fetch_feeds():
    """Download any feeds not already cached."""
    os.makedirs(BENCHMARK_DATA_DIR, exist_ok=True)
    to_fetch = [url for url in FEEDS if not os.path.exists(_cache_path(url))]
    if not to_fetch:
        print(f"All {len(FEEDS)} feeds already cached. Nothing to fetch.")
        return

    print(
        f"Fetching {len(to_fetch)} new feeds (skipping {len(FEEDS) - len(to_fetch)} cached)..."
    )
    fetched = 0
    failed = 0
    with httpx.Client(verify=False) as client:
        for url in to_fetch:
            path = _cache_path(url)
            try:
                resp = client.get(
                    url, timeout=20.0, follow_redirects=True, headers=HEADERS
                )
                with open(path, "wb") as f:
                    f.write(resp.content)
                fetched += 1
                print(f"  OK  {url} ({len(resp.content)} bytes)")
            except Exception as e:
                failed += 1
                print(f"  FAIL {url}: {e}")
    print(f"\nFetched {fetched} feeds, {failed} failed.")


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
        content = _load_content(url)

        # Test fastfeedparser
        try:
            times = []
            for _ in range(iterations):
                start_time = time.perf_counter()
                feed = fastfeedparser.parse(content)
                times.append(time.perf_counter() - start_time)
            result["ffp_time"] = statistics.median(times)
            result["ffp_entries"] = len(feed.entries)
            print(
                f"[{url}] FastFeedParser: {len(feed.entries)} entries in {result['ffp_time']:.3f}s (median of {iterations} runs)"
            )
        except Exception as e:
            result["ffp_time"] = statistics.median(times) if times else 0
            print(f"[{url}] FastFeedParser failed: {e}")

        # Test feedparser
        if not skip_feedparser:
            try:
                times = []
                for _ in range(iterations):
                    start_time = time.perf_counter()
                    feed = feedparser.parse(content)
                    times.append(time.perf_counter() - start_time)
                result["fp_time"] = statistics.median(times)
                result["fp_entries"] = len(feed.entries)
                print(
                    f"[{url}] Feedparser: {len(feed.entries)} entries in {result['fp_time']:.3f}s (median of {iterations} runs)"
                )
            except Exception as e:
                result["fp_time"] = statistics.median(times) if times else 0
                print(f"[{url}] Feedparser failed: {e}")

        if skip_feedparser:
            if result["ffp_time"] > 0:
                result["success"] = True
        else:
            if result["ffp_time"] > 0 and result["fp_time"] > 0:
                result["success"] = True
                print(f"[{url}] Speedup: {result['fp_time'] / result['ffp_time']:.1f}x")

    except Exception as e:
        print(f"[{url}] Failed to load feed: {e}")

    return result


def test_parsers(skip_feedparser=False, iterations=3):
    print(f"Testing feed parsers ({len(FEEDS)} feeds)...")
    if skip_feedparser:
        print("Running in FastFeedParser-only mode (-s)")
    cached = sum(1 for url in FEEDS if os.path.exists(_cache_path(url)))
    if cached == len(FEEDS):
        print("Using cached feed data from benchmark_data/")
    elif cached > 0:
        print(
            f"Using {cached} cached feeds, {len(FEEDS) - cached} will be fetched live"
        )
    else:
        print("No cached data — fetching live (run with --fetch to pre-download)")
    print(f"Each feed parsed {iterations} times, using median for timing")
    print("-" * 50)

    results = []
    overall_start_time = time.perf_counter()

    for url in FEEDS:
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

        if not skip_feedparser and r["ffp_entries"] != r["fp_entries"]:
            entry_mismatches.append(
                {
                    "url": r["url"],
                    "ffp_entries": r["ffp_entries"],
                    "fp_entries": r["fp_entries"],
                    "diff": r["ffp_entries"] - r["fp_entries"],
                }
            )

        if not skip_feedparser and r["fp_time"] > 0 and r["ffp_time"] > 0:
            speedup = r["fp_time"] / r["ffp_time"]
            if speedup < 1.1:
                slow_feeds.append(
                    {
                        "url": r["url"],
                        "speedup": speedup,
                        "ffp_time": r["ffp_time"],
                        "fp_time": r["fp_time"],
                    }
                )

    print("\nSummary:")
    print("-" * 50)
    print(f"Total wall-clock time: {overall_time:.2f}s")
    print(f"Successfully tested {successful_feeds}/{len(FEEDS)} feeds")
    if successful_feeds > 0:
        print("\nFastFeedParser:")
        print(f"  Total entries: {total_ffp_entries}")
        print(f"  Total parsing time: {total_ffp_time:.2f}s")
        print(f"  Average per feed: {total_ffp_time / successful_feeds:.3f}s")
        print(f"  Feeds/sec: {successful_feeds / total_ffp_time:.1f}")

        if not skip_feedparser:
            print("\nFeedparser:")
            print(f"  Total entries: {total_fp_entries}")
            print(f"  Total parsing time: {total_fp_time:.2f}s")
            print(f"  Average per feed: {total_fp_time / successful_feeds:.3f}s")
            print(f"  Feeds/sec: {successful_feeds / total_fp_time:.1f}")
            print(
                f"\nSpeedup: FastFeedParser is {(total_fp_time / total_ffp_time):.1f}x faster"
            )

            if entry_mismatches:
                print(
                    f"\nOUTLIERS: Entry Count Mismatches ({len(entry_mismatches)} feeds)"
                )
                print("-" * 50)
                for m in entry_mismatches:
                    print(f"  {m['url']}")
                    print(f"    FastFeedParser: {m['ffp_entries']} entries")
                    print(f"    Feedparser: {m['fp_entries']} entries")
                    print(f"    Difference: {m['diff']:+d}")

            if slow_feeds:
                print(
                    f"\nOUTLIERS: Slow Performance (<1.1x speedup, {len(slow_feeds)} feeds)"
                )
                print("-" * 50)
                slow_feeds.sort(key=lambda x: x["speedup"])
                for s in slow_feeds:
                    print(f"  {s['url']}")
                    print(f"    Speedup: {s['speedup']:.2f}x")
                    print(f"    FastFeedParser: {s['ffp_time'] * 1000:.2f}ms")
                    print(f"    Feedparser: {s['fp_time'] * 1000:.2f}ms")


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
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Pre-download all feeds into benchmark_data/",
    )
    args = parser.parse_args()

    if args.fetch:
        fetch_feeds()
    else:
        test_parsers(
            skip_feedparser=args.skip_feedparser,
            iterations=args.iterations,
        )
