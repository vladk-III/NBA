"""HTTP helper with retries, shared by every provider client."""

import logging
import time

import requests

LOG = logging.getLogger("nba_collector")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://www.nba.com/",
}


def get_json(url: str, params: dict | None = None, attempts: int = 4, timeout: int = 30,
             allow_404: bool = False) -> tuple[dict | list | None, requests.Response | None]:
    """GET a JSON document, retrying network errors, 429s and 5xx with backoff.

    Returns (payload, response). Payload is None when every attempt failed, or on a
    404 when allow_404 is set (e.g. a box score that is not published yet).
    """
    resp = None
    for attempt in range(attempts):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
            if resp.status_code == 404 and allow_404:
                return None, resp
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}", response=resp)
            resp.raise_for_status()
            return resp.json(), resp
        except (requests.RequestException, ValueError) as e:
            wait = 2 ** (attempt + 1)
            LOG.warning("GET %s failed (attempt %d/%d): %s", url.split("?")[0], attempt + 1, attempts, e)
            if attempt + 1 < attempts:
                time.sleep(wait)
    return None, resp
