"""HTML cache: key normalisation, round trip, expiry, robustness."""

import json
import re
from datetime import datetime, timedelta, timezone

from ebay_sold.fetch.cache import HtmlCache, cache_key, page_stem
from ebay_sold.models import SearchQuery
from ebay_sold.urls import search_url

BASE = "https://www.ebay.com/sch/i.html?_nkw=ta1+adapter&_sacat=0&LH_Sold=1&LH_Complete=1&_ipg=240"


class Clock:
    def __init__(self):
        self.now = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.now


def test_key_ignores_host_case_param_order_tracking_and_fragment():
    variants = [
        BASE,
        "https://WWW.EBAY.COM/sch/i.html?LH_Complete=1&_ipg=240&LH_Sold=1&_sacat=0&_nkw=ta1%20adapter",
        BASE + "&_trksid=p2334524.m570.l1313&_from=R40&_odkw=winsor&_osacat=0&itmmeta=01ABC&hash=item123#results",
        "https://www.ebay.com:443/sch/i.html?_nkw=ta1+adapter&_sacat=0&LH_Sold=1&LH_Complete=1&_ipg=240&mkevt=1",
    ]
    assert len({cache_key(u) for u in variants}) == 1
    assert len({page_stem(u) for u in variants}) == 1


def test_key_keeps_everything_that_changes_results():
    keys = {
        cache_key(BASE),
        cache_key(BASE + "&_pgn=2"),
        cache_key(BASE.replace("_ipg=240", "_ipg=60")),
        cache_key(BASE.replace("ta1+adapter", "ta2+adapter")),
        cache_key(BASE + "&LH_ItemCondition=3000"),
        cache_key(BASE + "&_sop=13"),
        cache_key(BASE.replace("www.ebay.com", "www.ebay.co.uk")),
    }
    assert len(keys) == 7


def test_stem_is_readable():
    url = search_url(SearchQuery(keywords="Hot Wheels R34 Zamac"), page=2)
    stem = page_stem(url)
    assert stem.startswith("hot-wheels-r34-zamac-p2-")
    assert page_stem(search_url(SearchQuery(keywords="Hot Wheels R34 Zamac"))).startswith("hot-wheels-r34-zamac-")
    assert page_stem("http://127.0.0.1:8000/sch/i.html").startswith("sch-i-html-")


def test_put_get_round_trip(tmp_path):
    clock = Clock()
    cache = HtmlCache(tmp_path / "html-cache", ttl_hours=24, now=clock)
    assert cache.get(BASE) is None
    html = "<html><body>café ⁣ \U0001f697 results</body></html>"
    path = cache.put(BASE, final_url=BASE + "&rt=nc", status=200, html=html)
    assert path == cache.path_for(BASE) and path.suffix == ".html"
    assert path.read_text(encoding="utf-8") == html
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["url"] == BASE and meta["status"] == 200 and meta["key"] == cache.key(BASE)
    assert sorted(p.name for p in path.parent.iterdir()) == sorted([path.name, path.with_suffix(".json").name])

    hit = cache.get(BASE + "&_trksid=abc")  # equivalent URL
    assert hit is not None
    assert hit.html == html and hit.status == 200 and hit.path == path
    assert hit.final_url == BASE + "&rt=nc"
    assert hit.fetched_at == clock.now


def test_expiry(tmp_path):
    clock = Clock()
    cache = HtmlCache(tmp_path, ttl_hours=24, now=clock)
    cache.put(BASE, final_url=BASE, status=200, html="<html></html>")
    clock.now += timedelta(hours=23, minutes=59)
    assert cache.get(BASE) is not None
    clock.now += timedelta(minutes=2)
    assert cache.get(BASE) is None

    forever = HtmlCache(tmp_path, ttl_hours=0, now=clock)
    clock.now += timedelta(days=365)
    assert forever.get(BASE) is not None


def test_put_overwrites(tmp_path):
    cache = HtmlCache(tmp_path, ttl_hours=24)
    cache.put(BASE, final_url=BASE, status=200, html="old")
    cache.put(BASE, final_url=BASE, status=200, html="new")
    assert cache.get(BASE).html == "new"
    assert len(list(tmp_path.glob("*.html"))) == 1


def test_broken_entries_are_misses(tmp_path):
    cache = HtmlCache(tmp_path, ttl_hours=24)
    path = cache.put(BASE, final_url=BASE, status=200, html="<html></html>")
    path.with_suffix(".json").write_text("{not json")
    assert cache.get(BASE) is None

    path = cache.put(BASE, final_url=BASE, status=200, html="<html></html>")
    path.unlink()  # metadata without html
    assert cache.get(BASE) is None

    path = cache.put(BASE, final_url=BASE, status=200, html="<html></html>")
    path.with_suffix(".json").unlink()  # html without metadata (interrupted write)
    assert cache.get(BASE) is None


def test_creates_missing_directory(tmp_path):
    cache = HtmlCache(tmp_path / "a" / "b", ttl_hours=1)
    assert cache.put(BASE, final_url=BASE, status=200, html="x").exists()


def test_marketing_params_are_dropped():
    for param in ("mkevt", "mkcid", "mkrid", "mksid", "mktype", "MKPID"):
        assert cache_key(BASE + f"&{param}=1") == cache_key(BASE), param
    assert cache_key(BASE + "&_mkt=1") != cache_key(BASE)  # only the prefix "mk", not any "mk" substring


def test_stem_is_a_safe_file_name_whatever_the_page_number(tmp_path):
    cache = HtmlCache(tmp_path / "cache", ttl_hours=24)
    for pgn in ("../../escape", "1/2", "9" * 300, "2%00x", "-1"):
        url = BASE + f"&_pgn={pgn}"
        stem = page_stem(url)
        assert re.fullmatch(r"[a-z0-9-]{1,80}", stem), stem
        path = cache.put(url, final_url=url, status=200, html="<html>ok</html>")
        assert path.parent == cache.dir and cache.get(url).html == "<html>ok</html>"
    assert page_stem(BASE + "&_pgn=12").startswith("ta1-adapter-p12-")
    assert len({page_stem(BASE + f"&_pgn={p}") for p in ("../../escape", "1/2")}) == 2  # the hash keeps them apart


def test_metadata_of_the_wrong_type_is_a_miss(tmp_path):
    cache = HtmlCache(tmp_path, ttl_hours=24)
    path = cache.put(BASE, final_url=BASE, status=200, html="<html></html>")
    for bad in ("null", "[]", '"str"', "5", '{"key": 5}'):
        path.with_suffix(".json").write_text(bad)
        assert cache.get(BASE) is None, bad


def test_truncated_or_edited_html_is_a_miss(tmp_path):
    cache = HtmlCache(tmp_path, ttl_hours=24)
    path = cache.put(BASE, final_url=BASE, status=200, html="<html>" + "x" * 1000 + "</html>")
    path.write_text("<html>xx")
    assert cache.get(BASE) is None
    path.write_bytes(b"\xff\xfe not utf-8")
    assert cache.get(BASE) is None


def test_entries_without_a_checksum_are_still_served(tmp_path):
    cache = HtmlCache(tmp_path, ttl_hours=24)
    path = cache.put(BASE, final_url=BASE, status=200, html="<html>old</html>")
    meta = json.loads(path.with_suffix(".json").read_text())
    del meta["sha1"]
    path.with_suffix(".json").write_text(json.dumps(meta))
    assert cache.get(BASE).html == "<html>old</html>"
