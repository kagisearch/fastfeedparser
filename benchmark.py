import time
import httpx
import feedparser
from fastfeedparser import parse as fastfeedparse

# Test feeds
feeds = [
    'https://alefesouza.com/feed/',
    'https://amitg.blog/feed/',
    'https://www.alwaystwisted.com/rss.php',
    'https://blog.kagi.com/rss.xml',
    'https://aaronfrancis.com/feed',
    'http://davidbau.com/index.rdf',
    'https://jesperbylund.com/rss',
    'https://aarvik.dk/rss/index.html',
    'http://dontcodetired.com/blog/syndication.axd',
    'http://www.coffeecoffeeandmorecoffee.com/atom.xml',
    'https://aivarsk.com/atom.xml',
    'http://ww25.abhijithpa.me/feed.xml?subid1=20241021-0621-269c-992a-93c4feb68a9f',
    'http://markcoddington.com/feed/',
    'https://andresb.net/blog/feed/',
    'https://albertgao.xyz/atom.xml',
    'http://annerallen.com/feed',
    'http://feeds.d15.biz/Daniel15',
    'https://alwaystwisted.com/rss.php',
    'https://aly.arriqaaq.com/rss/',
]

headers = {
    "User-Agent": "Mozilla/5.0 kagibot",
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
            start_time = time.time()
            try:
                feed = fastfeedparse(content)
                ffp_time = time.time() - start_time
                print(f"FastFeedParser: {len(feed.entries)} entries in {ffp_time:.3f}s")
                total_ffp_time += ffp_time
            except Exception as e:
                print(f"FastFeedParser failed: {e}")
                ffp_time = 0
            
            # Test feedparser
            start_time = time.time()
            try:
                feed = feedparser.parse(content)
                fp_time = time.time() - start_time
                print(f"Feedparser: {len(feed.entries)} entries in {fp_time:.3f}s")
                total_fp_time += fp_time
            except Exception as e:
                print(f"Feedparser failed: {e}")
                fp_time = 0
                
            if ffp_time > 0 or fp_time > 0:
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
