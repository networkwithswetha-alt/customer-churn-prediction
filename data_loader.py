"""Synthetic telecom customer churn dataset generator.

All data is generated in code with numpy/pandas. No real customer data is
used anywhere in this project.
"""

import numpy as np
import pandas as pd


def generate_churn_data(n_customers: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic telecom customer dataset with a churn label.

    The churn probability is a logistic function of a handful of plausible
    drivers (short tenure, month-to-month contract, high charges, no tech
    support, many support calls) plus random noise, so the data behaves
    like a realistic-but-fake churn problem.
    """
    rng = np.random.default_rng(seed)

    tenure = rng.integers(1, 73, size=n_customers)                      # months
    monthly_charges = rng.uniform(20.0, 120.0, size=n_customers).round(2)
    contract = rng.choice(
        ["month-to-month", "one-year", "two-year"],
        size=n_customers, p=[0.5, 0.3, 0.2],
    )
    internet = rng.choice(
        ["dsl", "fiber", "none"], size=n_customers, p=[0.4, 0.45, 0.15]
    )
    tech_support = rng.choice(["yes", "no"], size=n_customers, p=[0.35, 0.65])
    support_calls = rng.poisson(1.2, size=n_customers).clip(0, 10)
    paperless = rng.choice(["yes", "no"], size=n_customers, p=[0.6, 0.4])
    senior = rng.choice([0, 1], size=n_customers, p=[0.84, 0.16])

    total_charges = (tenure * monthly_charges * rng.uniform(0.9, 1.1, size=n_customers)).round(2)

    # Latent churn score -> probability via logistic function
    score = (
        -1.6
        + (-0.045 * tenure)
        + (0.018 * monthly_charges)
        + (support_calls * 0.35)
        + np.where(contract == "month-to-month", 0.9, 0.0)
        + np.where(contract == "one-year", -0.3, 0.0)
        + np.where(tech_support == "no", 0.5, 0.0)
        + np.where(internet == "fiber", 0.25, 0.0)
        + rng.normal(0, 0.6, size=n_customers)
    )
    prob = 1 / (1 + np.exp(-score))
    churn = (rng.random(n_customers) < prob).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract_type": contract,
            "internet_service": internet,
            "tech_support": tech_support,
            "support_calls": support_calls,
            "paperless_billing": paperless,
            "senior_citizen": senior,
            "churn": churn,
        }
    )


if __name__ == "__main__":
    df = generate_churn_data()
    print(df.head())
    print("\nChurn rate: {:.1%}".format(df["churn"].mean()))
    print("Rows:", len(df))
