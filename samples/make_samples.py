"""Generate the bundled synthetic datasets (seeded, so eval answers are stable). Run: python samples/make_samples.py"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent
rng = np.random.default_rng(42)

CITIES = {"Jakarta": 0.32, "Surabaya": 0.18, "Bandung": 0.15, "Medan": 0.12, "Semarang": 0.09, "Makassar": 0.08, "Denpasar": 0.06}
PRODUCTS = {  # category: [(product, base price IDR)]
    "Electronics": [("Earbuds", 350_000), ("Power Bank", 250_000), ("Smartwatch", 1_200_000)],
    "Fashion": [("Batik Shirt", 225_000), ("Sneakers", 650_000), ("Hijab", 85_000)],
    "Groceries": [("Coffee Beans 1kg", 180_000), ("Rice 5kg", 75_000), ("Cooking Oil 2L", 38_000)],
    "Home": [("Rice Cooker", 420_000), ("Fan", 310_000), ("Bedsheet Set", 195_000)],
}


def sales(n=3000) -> pd.DataFrame:
    days = pd.date_range("2025-01-01", "2026-06-30", freq="D")
    weights = np.linspace(1, 1.8, len(days))  # business grows over time -> visible trend for forecasting
    dates = rng.choice(days, n, p=weights / weights.sum())
    cats = rng.choice(list(PRODUCTS), n, p=[0.3, 0.3, 0.25, 0.15])
    rows = []
    for i, (d, c) in enumerate(zip(dates, cats)):
        product, base = PRODUCTS[c][rng.integers(3)]
        qty = int(rng.integers(1, 6))
        price = round(base * rng.uniform(0.9, 1.1), -2)
        disc = float(rng.choice([0, 0, 0, 5, 10, 15]))
        rows.append({
            "order_id": f"ORD-{10000 + i}", "order_date": pd.Timestamp(d).date().isoformat(),
            "city": rng.choice(list(CITIES), p=list(CITIES.values())), "channel": rng.choice(["Online", "Store"], p=[0.6, 0.4]),
            "category": c, "product": product, "quantity": qty, "unit_price": price, "discount_pct": disc,
            "revenue": round(qty * price * (1 - disc / 100)), "payment_method": rng.choice(["QRIS", "Bank Transfer", "E-Wallet", "Cash", "Credit Card"]),
            "customer_age": int(np.clip(rng.normal(31, 9), 17, 70)),
        })
    df = pd.DataFrame(rows).sort_values("order_date").reset_index(drop=True)
    # deliberate dirt so cleaning is worth demoing
    df.loc[rng.choice(len(df), 45, replace=False), "unit_price"] = np.nan
    df.loc[rng.choice(len(df), 30, replace=False), "customer_age"] = np.nan
    df.loc[rng.choice(len(df), 6, replace=False), "quantity"] = 250  # bulk-order outliers
    df["revenue"] = np.where(df["quantity"] == 250, (df["quantity"] * df["unit_price"].fillna(100_000)).round(), df["revenue"])
    return pd.concat([df, df.sample(25, random_state=1)]).reset_index(drop=True)


def employees(n=400) -> pd.DataFrame:
    depts = {"Engineering": 1.4, "Data": 1.3, "Product": 1.25, "Marketing": 1.0, "Sales": 0.95, "Finance": 1.05, "HR": 0.9}
    d = rng.choice(list(depts), n)
    exp = np.clip(rng.gamma(2.2, 2.2, n), 0, 25).round(1)
    salary = (6_500_000 + exp * 1_100_000) * np.array([depts[x] for x in d]) * rng.uniform(0.85, 1.15, n)
    perf = np.clip(rng.normal(3.4, 0.7, n), 1, 5).round(1)
    left = rng.random(n) < (0.28 - 0.04 * perf + 0.05 * (exp < 2))
    df = pd.DataFrame({
        "employee_id": [f"EMP-{i:04d}" for i in range(n)], "department": d,
        "city": rng.choice(["Jakarta", "Bandung", "Surabaya", "Remote"], n, p=[0.55, 0.15, 0.15, 0.15]),
        "hire_date": (pd.Timestamp("2026-06-30") - pd.to_timedelta((exp * 365 + rng.integers(0, 200, n)).astype(int), "D")).date.astype(str),
        "years_experience": exp, "monthly_salary_idr": (salary / 50_000).round() * 50_000,
        "performance_score": perf, "training_hours": rng.integers(0, 80, n), "left_company": np.where(left, "Yes", "No"),
    })
    df.loc[rng.choice(n, 18, replace=False), "performance_score"] = np.nan
    return df


if __name__ == "__main__":
    sales().to_csv(OUT / "sales_id.csv", index=False)
    employees().to_csv(OUT / "employees.csv", index=False)
    print("wrote", OUT / "sales_id.csv", OUT / "employees.csv")
