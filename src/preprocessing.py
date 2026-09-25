"""Keep original fields and add deterministic normalized fields."""
import pandas as pd
from .normalization import extract_address_components, normalize_address, normalize_business_name, normalize_text, raw_text
from .contacts import extract_contacts


def preprocess(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["name_norm"] = result.business_name.map(normalize_business_name)
    result["address_norm"] = result.business_address.map(normalize_address)
    result["country_norm"] = result.country.map(normalize_text)
    components = [extract_address_components(x) for x in result.business_address]
    for field in ("postal_code", "city", "state", "street_number", "street_tokens"):
        result[field] = [x[field] for x in components]
    contacts = [extract_contacts(raw_text(name) + " " + raw_text(address)) for name, address in zip(result.business_name, result.business_address)]
    for field in ("email", "email_domain", "website_domain", "phone"):
        result[field] = [x[field] for x in contacts]
    return result.reset_index(drop=True)
