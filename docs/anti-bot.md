# CAPTCHAs, bot flags, and how this version avoids them

The 2021 version got challenged constantly. This page explains why, what the
new fetcher does differently, and how to run it so challenges stay rare.

## What eBay looks at

eBay's search pages sit behind Akamai's bot manager plus eBay's own challenge
pages (`/splashui/challenge`, "Pardon Our Interruption", hCaptcha). Roughly,
they score each visitor on:

- **IP reputation.** Datacenter and cloud IPs, and anything on a public proxy
  list, start with a bad score. From a cloud machine, `www.ebay.com` answered
  every request with an Akamai `403 Error Page`, including requests from a
  real Chromium. Nothing in the browser can fix a flagged IP.
- **Consistency.** Does the user agent match the browser engine, the TLS
  handshake, client hints, fonts, time zone and language?
- **History.** Does the visitor have cookies from earlier visits, or is every
  request a brand-new visitor landing straight on a search URL?
- **Behaviour and volume.** Request rate, regularity, and how many pages a
  session loads.

## What the 2021 code did that triggered challenges

| 2021 code | Why it was flagged |
|---|---|
| `random-useragent` picked a **Firefox** UA and put it on **Chromium** | The UA string, JS engine features and client hints disagreed. That mismatch is a strong bot signal. |
| Free proxies scraped from `hidemy.name` | Free-proxy IPs are shared with abusers and already blocklisted. |
| A new headless browser per page, with an empty profile | Every request looked like a first-time visitor deep-linking to search results. |
| Default 60 results per page | Four times as many page loads as needed. |
| Fixed `waitForTimeout(5000)`, no pacing | Requests came in regular bursts. |

## What the fetcher does now

| Now | Effect |
|---|---|
| One **persistent browser profile** (`data/browser-profile`) | Cookies and history survive between runs, so you look like a returning visitor. |
| The browser's **own user agent**, with no UA override or fingerprint plugins | Nothing to mismatch. Set `channel = "chrome"` to drive your installed Google Chrome. |
| Fixed locale, time zone, and viewport | A stable fingerprint, and reproducible screenshots too. |
| **240 results per page** (`_ipg=240`) | A quarter of the page loads. |
| **HTML cache** (`data/html-cache`, 24 h TTL) | The same page is never fetched twice, and re-parsing is offline. |
| **Human pacing**: 6–18 s jittered delays, a 45–120 s break every 8 pages, 25 pages per run max | No bursts and no regular intervals. |
| Gradual scrolling with variable steps | Loads lazy content the way a reader would. |
| One home-page visit before the first search | Arrives the way people do. |
| **Challenge detection** that stops the run instead of retrying | Hammering a server that has started to doubt you is what turns a challenge into a block. |
| **Headed by default.** On a challenge it pauses and waits for you to solve it in the window | A real human answers the rare CAPTCHA. |

## How to run it so challenges stay rare

1. **Run it on your own computer and home connection**, not a cloud VM or a
   free proxy. This matters more than everything else combined.
2. Use your installed Chrome and a visible window (the default):

   ```toml
   # ebay-sold.toml
   [browser]
   channel = "chrome"
   headless = false
   ```

3. Warm up the profile once. Run `ebay-sold browser`, browse eBay normally for
   a minute, accept the cookie banner, and optionally sign in. Then close the
   window.
4. Keep the pacing defaults. For many queries, spread them over several runs
   or days. eBay keeps 90 days of sold history, so there's no rush.
5. **When a challenge appears, solve it in the window** and the run continues.
   If you see challenges on back-to-back runs, stop for the day. Your IP's
   score recovers with time, not with retries.
6. Use proxies only if you have to. If you do, use one sticky residential IP
   in your own country and time zone. Never rotate through free proxy lists.

### Zero-risk mode: let your own browser do the fetching

Open the sold search in your normal browser, press **Ctrl+S** / **Cmd+S**
("Webpage, HTML only" is enough), and import the saved files:

```bash
ebay-sold import-html ~/Downloads/*.html --keywords "hot wheels r34 zamac"
```

The same parser handles them, with no automation involved.

## What this project deliberately does not do

It does not integrate CAPTCHA-solving services, spoof browser fingerprints, or
rotate identities. Those are an arms race that eBay's terms forbid, and the
volumes a personal price tracker needs don't require them.

## The official alternative

- The Finding API (`findCompletedItems`) that people used for sold prices
  [was decommissioned on February 5, 2025](https://developer.ebay.com/develop/apis/api-deprecation-status).
- Its replacement, the
  [Marketplace Insights API](https://developer.ebay.com/api-docs/buy/marketplace-insights/overview.html)
  (`item_sales/search`), returns sold items for the last 90 days. It is a
  limited release that needs eBay's business approval, and it doesn't cover
  every category. If you can get access, it is the most robust source and can
  feed the same database.
- The Browse API only covers **active** listings, so it can't give sold prices.

## Terms of use

eBay's User Agreement restricts automated access to its site. Keep this to
personal research volumes, don't republish the data, and read the current
terms yourself. This is not legal advice.
