"""Temporary diagnostic for public StockAnalysis history-table pagination."""
from __future__ import annotations

from html.parser import HTMLParser
from io import StringIO

import pandas as pd
import requests


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.href = None
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self.href = dict(attrs).get("href")
            self.parts = []

    def handle_data(self, data):
        if self.href is not None:
            self.parts.append(data.strip())

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self.href is not None:
            self.links.append((" ".join(x for x in self.parts if x), self.href))
            self.href = None
            self.parts = []


def main():
    url = "https://stockanalysis.com/quote/bvc/ECOPETROL/history/"
    response = requests.get(
        url, timeout=20,
        headers={"User-Agent": "Mozilla/5.0 VFI-Investment-Analyzer research"},
    )
    print("STATUS", response.status_code, "FINAL_URL", response.url, "BYTES", len(response.text))
    response.raise_for_status()
    parser = Links()
    parser.feed(response.text)
    for label, href in parser.links:
        low = f"{label} {href}".lower()
        if any(token in low for token in ("next", "page=", "?p=", "offset=", "history")) or label in {"1", "2", "3", "›", "»"}:
            print("LINK", repr(label), href)
    try:
        tables = pd.read_html(StringIO(response.text))
        for i, table in enumerate(tables):
            print("TABLE", i, "SHAPE", table.shape, "COLUMNS", list(table.columns))
            if len(table):
                print(table.head(2).to_string(index=False))
    except Exception as exc:
        print("READ_HTML_ERROR", type(exc).__name__, str(exc))


if __name__ == "__main__":
    main()
