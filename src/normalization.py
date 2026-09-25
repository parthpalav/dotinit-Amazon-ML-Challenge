"""Conservative Unicode normalization and text-only address hints."""
import re
import unicodedata
import pandas as pd

NAME_WORDS = {"corp": "corporation", "intl": "international", "svc": "services", "svcs": "services"}
LEGAL_ENDINGS = ("private limited", "pvt ltd", "pvt limited", "private ltd", "limited", "ltd", "incorporated", "inc", "llc", "llp", "plc")
ADDRESS_WORDS = {"road": "rd", "street": "st", "avenue": "ave", "boulevard": "blvd",
                 "drive": "dr", "lane": "ln", "suite": "ste", "apartment": "apt"}


def raw_text(text) -> str:
    if text is None or pd.isna(text):
        return ""
    return str(text).strip()


def normalize_text(text) -> str:
    value = unicodedata.normalize("NFKC", raw_text(text)).casefold().replace("&", " and ")
    value = re.sub(r"['’`ʼ]", "", value)
    value = "".join(c if c.isalnum() or c.isspace() or unicodedata.category(c).startswith("M") else " " for c in value)
    return " ".join(value.split())


def normalize_business_name(text) -> str:
    value = " ".join(NAME_WORDS.get(t, t) for t in normalize_text(text).split())
    while value:
        previous = value
        for suffix in LEGAL_ENDINGS:
            if value.endswith(" " + suffix):
                value = value[: -len(suffix)].strip()
                break
        if value == previous:
            break
    return value


def normalize_address(text) -> str:
    return " ".join(ADDRESS_WORDS.get(t, t) for t in normalize_text(text).split())


def extract_address_components(text) -> dict[str, str]:
    """Hints, not verified geographic facts; unknown formats remain missing.

    Postal hints use trailing numeric/alphanumeric patterns. City/state hints
    require comma-separated segments. No country-specific branches or gazetteer.
    """
    raw = unicodedata.normalize("NFKC", raw_text(text)).casefold()
    postal = ""
    # Handle separated alphanumeric postal codes and numeric postal suffixes.
    patterns = (r"\b([a-z]\d[a-z]\s?\d[a-z]\d)\s*$",
                r"\b([a-z]{1,2}\d[a-z\d]?\s*\d[a-z]{2})\s*$",
                r"\b(\d{4,10}(?:-\d{3,4})?)\s*$")
    address_without_postal = raw
    for pattern in patterns:
        match = re.search(pattern, raw)
        if match:
            postal = re.sub(r"[\s-]", "", match.group(1))
            address_without_postal = raw[:match.start()].rstrip(" ,")
            break
    parts = [normalize_text(p) for p in address_without_postal.split(",") if normalize_text(p)]
    city = parts[-2] if len(parts) >= 3 else parts[-1] if len(parts) == 2 else ""
    state = parts[-1] if len(parts) >= 3 else ""
    number = re.match(r"\s*(\d+[a-z]?(?:[-/]\d+[a-z]?)?)\b", raw)
    normalized = normalize_address(raw)
    return {"postal_code": postal, "city": city, "state": state,
            "street_number": number.group(1) if number else "",
            "street_tokens": " ".join(t for t in normalized.split() if not t.isdigit())}


def ngrams(value: str, n: int) -> set[str]:
    value = value.replace(" ", "")
    if not value:
        return set()
    return {value[i:i+n] for i in range(max(1, len(value) - n + 1))}
