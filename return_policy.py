from datetime import date, datetime, timedelta
from typing import Dict, Optional

# Sourced from returns_and_refunds.md -> "General Return Window"
RETURN_WINDOW_DAYS = {
    "Electronics": 3,
}
NON_RETURNABLE_CATEGORIES = {"Cosmetics"}  # also Innerwear/perishables, not present in orders.csv
DEFAULT_WINDOW_DAYS = 7


def check_return_eligibility(order_details: Dict, today: Optional[date] = None) -> Dict:
    """Deterministically compute return eligibility. No LLM involved."""
    today = today or date.today()
    category = order_details.get("category", "")
    status = order_details.get("status", "")
    order_date = datetime.strptime(str(order_details["order_date"]), "%Y-%m-%d").date()

    if category in NON_RETURNABLE_CATEGORIES:
        return {
            "category": category,
            "status": status,
            "order_date": order_date,
            "window_days": None,
            "deadline": None,
            "eligible": False,
            "reason": f"{category} items are non-returnable for hygiene reasons.",
        }

    window_days = RETURN_WINDOW_DAYS.get(category, DEFAULT_WINDOW_DAYS)
    deadline = order_date + timedelta(days=window_days)
    eligible = (status == "Delivered") and (today <= deadline)

    reason = None
    if status != "Delivered":
        reason = f"Order status is '{status}', not yet Delivered, so the return window hasn't started."
    elif today > deadline:
        reason = f"The {window_days}-day return window closed on {deadline}."

    return {
        "category": category,
        "status": status,
        "order_date": order_date,
        "window_days": window_days,
        "deadline": deadline,
        "eligible": eligible,
        "reason": reason,
    }