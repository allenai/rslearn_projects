from datetime import datetime

def parse_name_as_date(name: str):
    """
    Parses Name values like '5/3/015' or '5/4/15' into datetime.
    Assumes M/D/YYYY where:
        - 15 → 2015
        - 015 → 2015
    """
    if not isinstance(name, str):
        return None

    parts = name.split("/")
    if len(parts) != 3:
        return None

    month_str, day_str, year_str = parts

    # Extract digits from year (handles '015', '15', etc.)
    year_digits = "".join(ch for ch in year_str if ch.isdigit())
    if not year_digits:
        return None

    y = int(year_digits)
    if y < 100:
        year = 2000 + y  # 15 → 2015
    else:
        year = 2000 + y  # 015 → 2015

    try:
        return datetime(year, int(month_str), int(day_str))
    except ValueError:
        return None
