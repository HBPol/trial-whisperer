# Minimal intent router placeholder (extend later)
def route_intent(query: str) -> str:
    q = query.lower()
    if "compare" in q and "nct" in q:
        return "compare"
    if "eligible" in q or "eligibility" in q:
        return "eligibility"
    return "qa"