"""HTTP helper with retries, shared by every provider client."""

import logging
import time
from urllib.parse import urlparse

import requests

LOG = logging.getLogger("nba_collector")

BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")

# cdn.nba.com answers 403 unless the request looks like it came from nba.com.
# This mirrors the header set nba_api's live client sends.
NBA_CDN_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate",
    "Accept-Language": "en-US,en;q=0.9",
    "Cache-Control": "max-age=0",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "User-Agent": BROWSER_UA,
}

DEFAULT_HEADERS = {"Accept": "application/json", "User-Agent": BROWSER_UA}


def headers_for(url: str) -> dict:
    host = urlparse(url).hostname or ""
    return NBA_CDN_HEADERS if host.endswith("nba.com") else DEFAULT_HEADERS


def get_json(url: str, params: dict | None = None, attempts: int = 4, timeout: int = 30,
             allow_404: bool = False, headers: dict | None = None,
             ) -> tuple[dict | list | None, requests.Response | None]:
    """GET a JSON document, retrying network errors, 429s and 5xx with backoff.

    Other 4xx responses (403 forbidden, 401, ...) are not retried: they will not
    change on a retry. Returns (payload, response); payload is None on failure,
    or on a 404 when allow_404 is set (e.g. a box score not published yet).
    """
    resp = None
    for attempt in range(attempts):
        try:
            resp = requests.get(url, params=params, headers=headers or headers_for(url), timeout=timeout)
            if resp.status_code == 404 and allow_404:
                return None, resp
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                LOG.warning("GET %s refused: HTTP %d", url.split("?")[0], resp.status_code)
                return None, resp
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            return resp.json(), resp
        except (requests.RequestException, ValueError) as e:
            LOG.warning("GET %s failed (attempt %d/%d): %s", url.split("?")[0], attempt + 1, attempts, e)
            if attempt + 1 < attempts:
                time.sleep(2 ** (attempt + 1))
    return None, resp
