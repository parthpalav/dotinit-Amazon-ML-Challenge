"""Extract explicit contact strings embedded in supplied text; no enrichment."""
import re
from urllib.parse import urlsplit

EMAIL = re.compile(r"(?<![\w.+-])([\w.!#$%&'*+/=?^`{|}~-]+)@([\w-]+(?:\.[\w-]+)+)", re.UNICODE)
URL = re.compile(r"(?:https?://|www\.)[^\s<>|]+", re.I)
PHONE = re.compile(r"\b(?:tel(?:ephone)?|phone|mobile|mob)\s*[:=]\s*(\+?[\d() .-]{7,25})", re.I)


def extract_contacts(text):
    text=str(text or "")
    email=EMAIL.search(text)
    # Preserve local-part casing; email local parts may be case-sensitive.
    email_value=email.group(1)+"@"+email.group(2).casefold() if email else ""
    url=URL.search(text);domain=""
    if url:
        value=url.group().rstrip(".,;)\"'")
        try:
            domain=(urlsplit(value if "://" in value else "https://"+value).hostname or "").casefold()
            if domain.startswith("www."):domain=domain[4:]
        except ValueError:
            domain=""
    phone=PHONE.search(text);phone_value=""
    if phone:
        raw=phone.group(1).strip();digits=re.sub(r"\D", "", raw)
        if 7<=len(digits)<=15:phone_value=("+" if raw.startswith("+") else "")+digits
    return {"email":email_value,"email_domain":email.group(2).casefold() if email else "",
            "website_domain":domain,"phone":phone_value}
