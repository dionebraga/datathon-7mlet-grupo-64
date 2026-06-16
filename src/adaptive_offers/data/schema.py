"""Schema of the Bank Marketing factual base and the processed layer.

We model the ``bank-additional-full.csv`` schema (henriqueyamahata / UCI Bank
Marketing). The single most important modelling decision in Stage 1 is the
explicit removal of **temporal-leakage** columns — chiefly ``duration``, which
is only known *after* the contact ends and therefore cannot be used by a
decision policy that runs *before* contacting the client.
"""

from __future__ import annotations

# --- Raw Bank Marketing columns (bank-additional-full.csv) -------------------

NUMERIC_COLUMNS: list[str] = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp_var_rate",
    "cons_price_idx",
    "cons_conf_idx",
    "euribor3m",
    "nr_employed",
]

CATEGORICAL_COLUMNS: list[str] = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

TARGET_COLUMN: str = "y"

# --- Leakage / post-contact columns (DROPPED before modelling) ---------------
# `duration`: last-contact duration in seconds; known only AFTER the call, and
# highly correlated with the target -> classic target leakage (documented in the
# UCI dataset notes themselves). We drop it from the decision feature set.
LEAKAGE_COLUMNS: list[str] = ["duration"]

# Categorical level vocabularies used by the deterministic facsimile generator
# so the offline base matches the real dataset's domain.
CATEGORY_LEVELS: dict[str, list[str]] = {
    "job": [
        "admin.", "blue-collar", "technician", "services", "management",
        "retired", "self-employed", "entrepreneur", "unemployed",
        "housemaid", "student", "unknown",
    ],
    "marital": ["married", "single", "divorced", "unknown"],
    "education": [
        "university.degree", "high.school", "basic.9y", "professional.course",
        "basic.4y", "basic.6y", "illiterate", "unknown",
    ],
    "default": ["no", "unknown", "yes"],
    "housing": ["yes", "no", "unknown"],
    "loan": ["no", "yes", "unknown"],
    "contact": ["cellular", "telephone"],
    "month": ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
    "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
    "poutcome": ["nonexistent", "failure", "success"],
}

# Columns that may act as sensitive proxies. They are NOT used as protected
# attributes for decisions; fairness is assessed over synthetic segments
# (Stage 4). Documented in docs/lgpd-plan.md.
SENSITIVE_PROXY_COLUMNS: list[str] = ["age", "job", "marital", "education"]

# `pdays == 999` is the dataset's sentinel for "client not previously contacted".
PDAYS_NOT_CONTACTED_SENTINEL: int = 999


def all_raw_columns() -> list[str]:
    """Full ordered list of raw columns including leakage + target."""
    return [
        "age", "job", "marital", "education", "default", "housing", "loan",
        "contact", "month", "day_of_week", "duration", "campaign", "pdays",
        "previous", "poutcome", "emp_var_rate", "cons_price_idx",
        "cons_conf_idx", "euribor3m", "nr_employed", "y",
    ]


def feature_columns() -> list[str]:
    """Modelling features = numeric + categorical, excluding leakage & target."""
    return NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
