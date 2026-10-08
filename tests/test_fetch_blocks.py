"""Challenge / block page detection (no browser)."""

from pathlib import Path

import pytest

from ebay_sold.fetch.blocks import detect_block, has_captcha_widget, is_ebay_host, looks_like_results, page_title

from conftest import FIXTURES, load_fixture

BLOCKS = FIXTURES / "blocks"
SEARCH = "https://www.ebay.com/sch/i.html?_nkw=hot+wheels&LH_Sold=1&LH_Complete=1&_ipg=240"


def block_page(name: str) -> str:
    return (BLOCKS / f"{name}.html").read_text(encoding="utf-8")


def test_real_results_pages_are_not_blocks(ebay_page):
    name, html = ebay_page
    assert looks_like_results(html), name
    assert detect_block(status=200, url=SEARCH, html=html) is None, name


def test_footer_that_mentions_captcha_is_not_a_block():
    html = load_fixture("sold_2026-04-16_ta1-adapter")
    footer = (
        '<footer><a href="https://www.ebay.com/help/account/captcha?id=4204">Why am I asked to complete a captcha?</a>'
        ' <a href="https://www.google.com/recaptcha/about/">About reCAPTCHA</a>'
        " <p>Security Measure: we never ask for your password by email. Access Denied? Contact us.</p></footer>"
    )
    html = html.replace("</body>", footer + "</body>")
    assert detect_block(status=200, url=SEARCH, html=html) is None


def test_results_page_title_that_contains_a_challenge_phrase():
    html = "<html><head><title>Access Denied Poster for sale | eBay</title></head><body><p>No results</p></body></html>"
    assert detect_block(status=200, url=SEARCH, html=html) is None


@pytest.mark.parametrize(
    ("fixture", "status", "url", "kind"),
    [
        ("akamai_403", 403, SEARCH, "access_denied"),
        ("access_denied", 403, SEARCH, "access_denied"),
        ("access_denied", 200, SEARCH, "access_denied"),  # title alone is enough
        ("splashui_challenge", 200, "https://www.ebay.com/splashui/challenge?ap=1&appName=orch&ru=x", "captcha"),
        ("splashui_challenge", 200, SEARCH, "captcha"),  # widget/title, even without the redirect
        ("splashui_challenge", 200, "https://www.ebay.com/splashui/captcha?ru=x", "captcha"),
        ("pardon_interruption", 200, SEARCH, "interstitial"),
        ("pardon_interruption", 405, SEARCH, "interstitial"),
        ("signin_redirect", 200, "https://signin.ebay.com/ws/eBayISAPI.dll?SignIn&ru=https%3A%2F%2Fwww.ebay.com",
         "signin"),
        ("signin_redirect", 200, "https://signin.ebay.co.uk/signin/?ru=x", "signin"),
    ],
)
def test_block_fixtures(fixture, status, url, kind):
    info = detect_block(status=status, url=url, html=block_page(fixture))
    assert info is not None
    assert info.kind == kind
    assert info.url == url and info.status == status
    assert info.reason


def test_akamai_error_page_needs_the_403():
    # "Error Page | eBay" with a 200 is just an error page, not an edge block.
    assert detect_block(status=200, url=SEARCH, html=block_page("akamai_403")) is None
    info = detect_block(status=403, url=SEARCH, html=block_page("akamai_403"))
    assert info.kind == "access_denied" and "Akamai" in info.reason
    assert info.title == "Error Page | eBay"


def test_rate_limited_regardless_of_body():
    info = detect_block(status=429, url=SEARCH, html=load_fixture("sold_2026-04-16_ta1-adapter"))
    assert info.kind == "rate_limited"


def test_any_403_is_access_denied():
    info = detect_block(status=403, url=SEARCH, html="<html><head><title>Forbidden</title></head></html>")
    assert info.kind == "access_denied"


def test_captcha_widgets():
    recaptcha_iframe = '<html><body><iframe src="https://www.google.com/recaptcha/api2/anchor?k=x"></iframe></body></html>'
    recaptcha_div = '<html><body><div class="g-recaptcha" data-sitekey="x"></div></body></html>'
    hcaptcha_script = '<html><head><script src="https://js.hcaptcha.com/1/api.js"></script></head></html>'
    for html in (recaptcha_iframe, recaptcha_div, hcaptcha_script):
        assert detect_block(status=200, url=SEARCH, html=html).kind == "captcha"


def test_sign_in_links_on_a_normal_page_are_not_a_redirect():
    html = (
        "<html><head><title>Hot Wheels for sale | eBay</title></head>"
        '<body><a href="https://signin.ebay.com/ws/eBayISAPI.dll?SignIn">Sign in</a></body></html>'
    )
    assert detect_block(status=200, url=SEARCH, html=html) is None


def test_challenge_path_is_detected_on_any_host():
    # the browser tests serve challenge pages from 127.0.0.1
    info = detect_block(status=200, url="http://127.0.0.1:8000/splashui/challenge", html="<html></html>")
    assert info.kind == "captcha"


def test_helpers():
    assert page_title("<html><head><title>  Ta1 &amp; Adapter\n | eBay</title>") == "Ta1 & Adapter | eBay"
    assert page_title("<html></html>") is None
    assert is_ebay_host("www.ebay.com") and is_ebay_host("signin.ebay.co.uk") and is_ebay_host("www.ebay.com.au")
    assert not is_ebay_host("127.0.0.1") and not is_ebay_host("notebay.com") and not is_ebay_host("ebay.com.evil.net")
    assert not looks_like_results(block_page("splashui_challenge"))


def test_fixture_files_exist():
    names = {p.stem for p in Path(BLOCKS).glob("*.html")}
    assert {"akamai_403", "splashui_challenge", "pardon_interruption", "access_denied", "signin_redirect"} <= names


def test_browser_check_wording_is_an_interstitial():
    html = ("<html><head><title>eBay</title></head><body><p>Checking your browser before you access eBay.</p>"
            "<p>Your browser will redirect to your requested content shortly.</p></body></html>")
    info = detect_block(status=200, url=SEARCH, html=html)
    assert info.kind == "interstitial" and not has_captcha_widget(html)


def test_has_captcha_widget():
    assert has_captcha_widget(block_page("splashui_challenge"))
    assert not has_captcha_widget(block_page("pardon_interruption"))
    assert not has_captcha_widget(load_fixture("sold_2026-04-16_ta1-adapter"))
    assert has_captcha_widget('<div class="note h-captcha"></div>')
