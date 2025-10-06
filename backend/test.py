def extract_transactions(raw_text: str) -> str:
    """
    Keep only lines that look like transactions:
    MM/DD description amount
    """
    pattern = r"(\d{2}/\d{2})\s+(.+?)\s+(-?\$?\d+\.\d{2})"
    matches = re.findall(pattern, raw_text)

    transactions = []
    for m in matches:
        date, merchant, amount = m
        # normalize amount (remove $ sign)
        amt = amount.replace("$", "")
        transactions.append(f"{date} {merchant.strip()} {amt}")
    return "\n".join(transactions)  
