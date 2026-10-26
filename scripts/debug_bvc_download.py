"""Temporary diagnostic for the official BVC historical XLS download endpoint."""
from __future__ import annotations

from datetime import date, timedelta
from io import BytesIO

import pandas as pd
import requests

ENDPOINTS = [
    "https://www.bvc.com.co/mercados/DescargaXlsServlet",
    "http://www.bvc.com.co/mercados/DescargaXlsServlet",
]


def main():
    end = date.today()
    start = end - timedelta(days=180)
    params = {
        "archivo": "acciones_detalle",
        "nemo": "ECOPETROL",
        "tipoMercado": "1",
        "fechaIni": start.isoformat(),
        "fechaFin": end.isoformat(),
    }
    for endpoint in ENDPOINTS:
        try:
            response = requests.get(
                endpoint, params=params, timeout=30,
                headers={"User-Agent": "Mozilla/5.0 VFI-Investment-Analyzer data validation"},
            )
            print("BVC_ENDPOINT", endpoint, "STATUS", response.status_code,
                  "FINAL_URL", response.url, "TYPE", response.headers.get("content-type"),
                  "BYTES", len(response.content), "MAGIC", repr(response.content[:16]))
            import re
            from html import unescape
            html_text = unescape(re.sub(r"<[^>]+>", " ", response.text))
            print("BVC_RESPONSE_TEXT", " ".join(html_text.split())[:1800])
            if response.ok and len(response.content) > 1000:
                try:
                    sheets = pd.read_excel(BytesIO(response.content), sheet_name=None, engine="xlrd")
                    for name, frame in sheets.items():
                        print("BVC_SHEET", name, "SHAPE", frame.shape, "COLUMNS", list(frame.columns))
                        print(frame.head(5).to_string(index=False))
                    return
                except Exception as exc:
                    print("BVC_EXCEL_PARSE_ERROR", type(exc).__name__, str(exc))
        except Exception as exc:
            print("BVC_REQUEST_ERROR", endpoint, type(exc).__name__, str(exc))


if __name__ == "__main__":
    main()
