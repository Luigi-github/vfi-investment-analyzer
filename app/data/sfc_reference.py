"""Free reference checks against the Superintendencia Financiera (SFC).

SFC publishes a weighted-average price and last-operation date, not full OHLCV
candles. This module therefore uses SFC only as a freshness/reference check and
never substitutes its PPP for the Close or the volume input to VFI.
"""
from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
import time
import unicodedata
from urllib.parse import urljoin

import pandas as pd
import requests

SFC_INDEX_URL = (
    "https://www.superfinanciera.gov.co/info/superfinancierav8/media/"
    "MercadoAccionario/Precios/Diarios/ppal_promedios.htm"
)
SFC_HINTS = {
    "ECOPETROL.CL": "AO PEC ECOPETROL S.A.",
    "CIBEST.CL": "AO GRUPO CIBEST S.A.",
    "ISA.CL": "AO INTERCONEXION ELECTRICA S.A. E.S.P.",
    "MINEROS.CL": "AO MINEROS S.A.",
    "GRUPOARGOS.CL": "AO GRUPO ARGOS S. A",
    "CEMARGOS.CL": "AO CEMENTOS ARGOS S.A",
    "NUTRESA.CL": "AO GRUPO NUTRESA S.A.",
    "BOGOTA.CL": "AO BANCO DE BOGOTA",
    "CELSIA.CL": "AO CELSIA S.A.",
    "CONCONCRET.CL": "AO CONCONCRETO S.A.",
    "CORFICOLCF.CL": "AO CORPORACION FINANCIERA COLOMBIANA S.A.",
    "EXITO.CL": "AO ALMACENES EXITO S.A.",
    "GEB.CL": "AO GRUPO ENERGIA BOGOTA S.A. E.S.P.",
    "GRUPOAVAL.CL": "AO PEC GRUPO AVAL ACCIONES Y VALORES S.A.",
    "GRUPOSURA.CL": "AO GRUPO DE INVERSIONES SURAMERICANA S.A.",
    "GRUBOLIVAR.CL": "AO GRUPO BOLIVAR S.A.",
    "PROMIGAS.CL": "AO PROMIGAS S.A. E.S.P.",
    "TERPEL.CL": "AO ORGANIZACION TERPEL S.A.",
    "ENKA.CL": "AO ENKA DE COLOMBIA S.A.",
    "ETB.CL": "AO EMPRESA DE TELECOMUNICACIONES DE BOGOTA S.A. ESP.",
    "PFCIBEST.CL": "ADP SIN DERECHO A VOTO GRUPO CIBEST S.A.",
    "PFAVAL.CL": "ADP PEC GRUPO AVAL ACCIONES Y VALORES",
    "PFCEMARGOS.CL": "ADP SIN DERECHO A VOTO CEMENTOS ARGOS S.A.",
    "PFCORFICOL.CL": "ADP CORPORACION FINANCIERA COLOMBIANA S.A.",
    "PFDAVVNDA.CL": "ADP SIN DERECHO A VOTO DAVIVIENDA GROUP S.A.",
    "PFGRUPOARG.CL": "DERECHOS ADP SIN DERECHO A DIVIDENDO GRUPO ARGOS ESPECIE SDPFGR",
    "PFGRUPSURA.CL": "ACCION SIN DIVIDENDO SURAMERICANA DE INVERSIONES S.A. SURAMERICA",
}


def _norm(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()


class _AnchorParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self._href = None
        self._parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self._href = dict(attrs).get("href")
            self._parts = []

    def handle_data(self, data):
        if self._href is not None:
            self._parts.append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self._href is not None:
            label = " ".join(" ".join(self._parts).split())
            if label:
                self.links.append((label, self._href))
            self._href = None
            self._parts = []


class _TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        if data.strip():
            self.parts.append(data.strip())


def parse_sfc_detail(html: str, symbol: str = "") -> dict:
    parser = _TextParser()
    parser.feed(html)
    text = re.sub(r"\s+", " ", " ".join(parser.parts))
    report = re.search(r"Fecha de reporte\s*:\s*(\d{2}/\d{2}/\d{4})", text, re.I)
    last_op = re.search(r"Fecha de la ultima operaci[oó]n\s*:\s*(\d{2}/\d{2}/\d{4})", text, re.I)
    section_match = re.search(r"PROMEDIO PONDERADO(.*?)Suspensiones", text, re.I)
    ppp = None
    if section_match:
        section = section_match.group(1)
        section = re.sub(r"\d{2}/\d{2}/\d{4}", " ", section)
        numbers = re.findall(r"(?<![A-Z])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", section, re.I)
        if numbers:
            try:
                ppp = float(numbers[-4].replace(",", "")) if len(numbers) >= 4 else float(numbers[0].replace(",", ""))
            except (ValueError, IndexError):
                ppp = None
    report_date = pd.to_datetime(report.group(1), dayfirst=True).date().isoformat() if report else None
    last_operation_date = pd.to_datetime(last_op.group(1), dayfirst=True).date().isoformat() if last_op else None
    return {
        "Symbol": symbol,
        "SFC_Report_Date": report_date,
        "SFC_Last_Operation_Date": last_operation_date,
        "SFC_PPP_COP": ppp,
        "SFC_Status": "OK" if report_date and last_operation_date and ppp is not None else "PARSE_INCOMPLETE",
    }


def _find_detail_url(symbol: str, links: list[tuple[str, str]]) -> str | None:
    hint = SFC_HINTS.get(symbol)
    if not hint:
        return None
    target = _norm(hint)
    candidates = [(label, href) for label, href in links if target in _norm(label)]
    if not candidates:
        # Fall back to the longest common prefix only when it is reasonably specific.
        target_words = target.split()
        candidates = [
            (label, href) for label, href in links
            if len(target_words) >= 3 and " ".join(_norm(label).split()[:len(target_words)]) == target
        ]
    return candidates[0][1] if candidates else None


def collect_sfc_references(
    symbols: list[str],
    latest_quotes: pd.DataFrame,
    output_dir: str | Path,
    *,
    timeout: int = 15,
    sleep_seconds: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch SFC issuer reference pages and compare their dates with Yahoo cache."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "VFI-Investment-Analyzer/1.0 (research; contact via repository)"})
    records = []
    try:
        response = session.get(SFC_INDEX_URL, timeout=timeout)
        response.raise_for_status()
        index_parser = _AnchorParser()
        index_parser.feed(response.text)
        index_links = index_parser.links
        index_error = ""
    except Exception as exc:
        index_links = []
        index_error = f"{type(exc).__name__}: {exc}"

    quotes = latest_quotes.set_index("Symbol", drop=False) if not latest_quotes.empty else pd.DataFrame()
    for symbol in symbols:
        record = {"Symbol": symbol, "SFC_Source_URL": SFC_INDEX_URL}
        if index_error:
            record.update({"SFC_Status": "SOURCE_UNAVAILABLE", "SFC_Error": index_error})
            records.append(record)
            continue
        href = _find_detail_url(symbol, index_links)
        if not href:
            record.update({"SFC_Status": "SYMBOL_NOT_MAPPED", "SFC_Error": "No matching SFC issuer page"})
            records.append(record)
            continue
        url = urljoin(SFC_INDEX_URL, href)
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            parsed = parse_sfc_detail(response.text, symbol=symbol)
            record.update(parsed)
            record["SFC_Source_URL"] = url
            record["SFC_Error"] = ""
        except Exception as exc:
            record.update({"SFC_Status": "SOURCE_UNAVAILABLE", "SFC_Source_URL": url, "SFC_Error": f"{type(exc).__name__}: {exc}"})
        if not quotes.empty and symbol in quotes.index:
            quote = quotes.loc[symbol]
            yahoo_date = pd.to_datetime(quote.get("Date"), errors="coerce")
            sfc_operation = pd.to_datetime(record.get("SFC_Last_Operation_Date"), errors="coerce")
            record["Yahoo_Last_Date"] = yahoo_date.date().isoformat() if pd.notna(yahoo_date) else None
            record["Yahoo_Close"] = quote.get("Close")
            if pd.notna(yahoo_date) and pd.notna(sfc_operation):
                record["SFC_Last_Operation_After_Yahoo"] = bool(sfc_operation.normalize() > yahoo_date.normalize())
                record["Yahoo_Date_vs_SFC_Operation_Days"] = int((yahoo_date.normalize() - sfc_operation.normalize()).days)
            else:
                record["SFC_Last_Operation_After_Yahoo"] = None
                record["Yahoo_Date_vs_SFC_Operation_Days"] = None
            ppp = record.get("SFC_PPP_COP")
            # PPP and Yahoo close are only compared numerically when both refer
            # to the same last-operation date; otherwise show no percentage.
            same_date = (
                pd.notna(yahoo_date)
                and pd.notna(sfc_operation)
                and yahoo_date.normalize() == sfc_operation.normalize()
            )
            if same_date and pd.notna(quote.get("Close")) and ppp and ppp > 0:
                record["Yahoo_Close_vs_SFC_PPP_Pct"] = round((float(quote["Close"]) / float(ppp) - 1) * 100, 4)
            else:
                record["Yahoo_Close_vs_SFC_PPP_Pct"] = None
        records.append(record)
        if sleep_seconds:
            time.sleep(sleep_seconds)

    reference = pd.DataFrame(records)
    reference_path = output_dir / "sfc_reference.csv"
    reference.to_csv(reference_path, index=False)
    comparison = reference.copy()
    if not comparison.empty and "SFC_Last_Operation_After_Yahoo" in comparison.columns:
        comparison["Comparison_Flag"] = comparison["SFC_Last_Operation_After_Yahoo"].map(
            {True: "YAHOO_DATE_BEHIND_SFC_OPERATION", False: "NO_DATE_CONFLICT"}
        ).fillna("UNVERIFIED")
    comparison.to_csv(output_dir / "source_comparison.csv", index=False)
    return reference, comparison
