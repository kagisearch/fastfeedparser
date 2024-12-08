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
    "https://lisacharlottemuth.com/atom.xml", 
    "https://emacsninja.com/feed.atom",
    "http://causality.cs.ucla.edu/blog/index.php/feed/",
    "https://blog.railsapps.org/rss",
    "https://www.bravelysheblogs.com/feed/",
    "https://realphysics.blogspot.com/feeds/posts/default",
    "https://staysaasy.com/feed.xml",
    "https://explog.in/rss.xml",
    "https://www.petekeen.net/index.xml",
    "https://jelleraaijmakers.nl/feed",
    "https://fale.io/index.xml",
    "https://gessfred.xyz/rss.xml",
    "https://fanf.dreamwidth.org/data/rss",
    "https://jacobwsmith.xyz/feed.xml",
    "https://bernsteinbear.com/feed.xml",
    "https://alefesouza.com/feed/",
    "https://amitg.blog/feed/",
    "https://www.alwaystwisted.com/rss.php",
    "https://blog.kagi.com/rss.xml",
    "https://aaronfrancis.com/feed",
    "http://davidbau.com/index.rdf",
    "https://jesperbylund.com/rss",
    "https://aarvik.dk/rss/index.html",
    "http://dontcodetired.com/blog/syndication.axd",
    "http://www.coffeecoffeeandmorecoffee.com/atom.xml",
    "https://aivarsk.com/atom.xml",
    "http://ww25.abhijithpa.me/feed.xml?subid1=20241021-0621-269c-992a-93c4feb68a9f",
    "http://markcoddington.com/feed/",
    "https://andresb.net/blog/feed/",
    "http://feeds.d15.biz/Daniel15",
    "https://alwaystwisted.com/rss.php",
    "https://aly.arriqaaq.com/rss/",
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
]

headers = {
    "User-Agent": "Mozilla/5.0 fastfeedparser",
    "Accept": "*/*",
    "Connection": "close",
}

client = httpx.Client(verify=False)


def test_parsers():
    print("Testing feed parsers...")
    print("-" * 50)

    total_ffp_time = 0
    total_fp_time = 0
    successful_feeds = 0

    for url in feeds:
        print(f"\nTesting {url}")
        try:
            resp = client.get(url, timeout=20.0, follow_redirects=True, headers=headers)
            content = resp.content

            # Test fastfeedparser
            try:
                start_time = time.perf_counter()
                feed = fastfeedparser.parse(content)
                ffp_time = time.perf_counter() - start_time
                print(f"FastFeedParser: {len(feed.entries)} entries in {ffp_time:.3f}s")
            except Exception as e:
                ffp_time = time.perf_counter() - start_time
                print(f"FastFeedParser failed: {e}")

            # Test feedparser
            try:
                start_time = time.perf_counter()
                feed = feedparser.parse(content)
                fp_time = time.perf_counter() - start_time
                print(f"Feedparser: {len(feed.entries)} entries in {fp_time:.3f}s")                
            except Exception as e:
                fp_time = time.perf_counter() - start_time
                print(f"Feedparser failed: {e}")

            total_ffp_time += ffp_time
            total_fp_time += fp_time    

            print(f"Speedup: {fp_time/ffp_time:.1f}x")
            if ffp_time > 0 and fp_time > 0:
                successful_feeds += 1

        except Exception as e:
            print(f"Failed to fetch feed: {e}")

    print("\nSummary:")
    print("-" * 50)
    print(f"Successfully tested {successful_feeds} feeds")
    if successful_feeds > 0:
        print(f"Average FastFeedParser time: {total_ffp_time/successful_feeds:.3f}s")
        print(f"Average Feedparser time: {total_fp_time/successful_feeds:.3f}s")
        print(f"FastFeedParser is {(total_fp_time/total_ffp_time):.1f}x faster")


if __name__ == "__main__":
    test_parsers()
