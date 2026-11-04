from bs4 import BeautifulSoup

# Matched against script src and content, link href and the generator meta tag,
# so a page that only mentions a product in its text does not count.
SIGNATURES = {
    "Shopify": ["cdn.shopify.com", "shopify.theme"],
    "WordPress": ["/wp-content/", "/wp-includes/", "generator:wordpress"],
    "WooCommerce": ["/plugins/woocommerce/", "wc-add-to-cart", "generator:woocommerce"],
    "HubSpot": ["js.hs-scripts.com", "js.hs-analytics.net", "js.hsforms.net"],
    "Google Analytics": ["googletagmanager.com/gtag/js", "google-analytics.com/analytics.js"],
    "Google Tag Manager": ["googletagmanager.com/gtm.js"],
    "Meta Pixel": ["fbevents.js"],
    "LinkedIn Insight Tag": ["snap.licdn.com"],
    "Webflow": ["website-files.com", "generator:webflow"],
    "Wix": ["static.wixstatic.com", "generator:wix"],
    "Squarespace": ["static1.squarespace.com", "generator:squarespace"],
    "Klaviyo": ["static.klaviyo.com"],
    "Stripe": ["js.stripe.com"],
    "Segment": ["cdn.segment.com"],
    "Intercom": ["widget.intercom.io"],
    "Hotjar": ["static.hotjar.com"],
    "Marketo": ["munchkin.marketo.net"],
    "Pardot": ["pi.pardot.com"],
    "Next.js": ["/_next/static/"],
}


def detect_tech(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    parts: list[str] = []
    for script in soup.find_all("script"):
        parts.append(script.get("src") or "")
        parts.append(script.string or "")
    for link in soup.find_all("link"):
        parts.append(link.get("href") or "")
    for meta in soup.find_all("meta", attrs={"name": "generator"}):
        parts.append(f"generator:{meta.get('content') or ''}")
    haystack = "\n".join(parts).lower()

    return [
        tech
        for tech, signatures in SIGNATURES.items()
        if any(signature in haystack for signature in signatures)
    ]
