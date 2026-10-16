
import re
import math


def deterministic_business_answer(question, documents, metadatas):
    """
    Handle highest/lowest regional revenue questions using Python.

    Returns an answer if enough comparable regional figures are found.
    Returns None when the information is insufficient or contradictory.
    """

    q = question.casefold()

    asks_highest = any(
        word in q
        for word in ("highest", "largest", "most")
    )

    asks_lowest = any(
        word in q
        for word in ("lowest", "smallest", "least")
    )

    if not (
        "region" in q
        and ("revenue" in q or "sales" in q)
        and (asks_highest or asks_lowest)
    ):
        return None

    # Supports sentences such as:
    # "The North region generated $190,000 in revenue."
    pattern = re.compile(
        r"\b(?:the\s+)?"
        r"(?P<region>[A-Za-z][A-Za-z &'’-]*?)"
        r"\s+region\s+generated\s+"
        r"\$?\s*(?P<amount>\d[\d,]*(?:\.\d+)?)"
        r"\s+in\s+revenue\b",
        re.IGNORECASE,
    )

    values = {}

    for document, metadata in zip(documents, metadatas):
        for match in pattern.finditer(document):
            region = " ".join(match.group("region").split())
            amount = float(
                match.group("amount").replace(",", "")
            )

            key = region.casefold()

            # Conflicting numbers for one region make the comparison
            # unsafe unless the data has first been separated by period.
            if key in values:
                if not math.isclose(
                    values[key]["amount"],
                    amount,
                    rel_tol=0,
                    abs_tol=0.000001,
                ):
                    return None
            else:
                values[key] = {
                    "region": region,
                    "amount": amount,
                    "source": metadata.get("source", "Unknown"),
                    "page": metadata.get(
                        "page",
                        metadata.get("chunk", "Unknown"),
                    ),
                }

    if len(values) < 2:
        return None

    records = list(values.values())

    target = (
        max(item["amount"] for item in records)
        if asks_highest
        else min(item["amount"] for item in records)
    )

    winners = [
        item for item in records
        if math.isclose(
            item["amount"],
            target,
            rel_tol=0,
            abs_tol=0.000001,
        )
    ]

    amount_text = (
        f"${target:,.2f}"
        if not target.is_integer()
        else f"${target:,.0f}"
    )

    rank = "highest" if asks_highest else "lowest"

    if len(winners) > 1:
        names = ", ".join(
            item["region"] for item in winners
        )

        return (
            f"The following regions tie for the {rank} recorded "
            f"revenue at {amount_text}: {names}. "
            "The available evidence does not identify a unique winner."
        )

    winner = winners[0]

    return (
        f"The {winner['region']} region generated {amount_text} "
        f"in revenue, the {rank} among the regions found in the "
        f"indexed documents. "
        f"[Source: {winner['source']}, "
        f"page/chunk: {winner['page']}]"
    )
