# Customer Churn Prediction

> Portfolio recreation with synthetic data, inspired by professional experience.

## The business problem

Acquiring a new telecom customer costs far more than keeping an existing one.
If we can flag customers who are likely to cancel **before** they leave, the
retention team can intervene — with a better plan, a support call, or a loyalty
offer — instead of reacting after the fact.

This project builds that early-warning system on a synthetic customer dataset:
given tenure, charges, contract type, and support history, predict whether a
customer will churn.

## What the model found

Running `python train.py` trains two classifiers and compares them:

- **Logistic Regression** — a transparent baseline; every coefficient reads as
  "how much this factor moves churn risk," which is exactly what a business
  stakeholder wants to see first.
- **Random Forest** — captures non-linear interactions (e.g., *new* customers
  on *month-to-month* contracts with *many support calls* are the riskiest
  segment) and usually scores higher on recall.

The `images/` folder holds the confusion matrix for each model and a feature
importance chart. In this synthetic data, the strongest churn signals are
**short tenure**, **month-to-month contracts**, **high monthly charges**,
**no tech support**, and **repeated support calls** — the same patterns
retention teams see in the real world.

For a retention program, **recall on the churn class matters more than raw
accuracy**: missing a would-be churner costs more than calling a happy
customer. That's why the report prints precision/recall/F1 per class, not
just one accuracy number.

## Limitations

- The data is **synthetic** — generated in `data_loader.py` from plausible
  assumptions, not real customer records. Treat every number as illustrative.
- No time dimension: real churn modeling uses behavioral trends over months.
- Class imbalance is mild here; production systems often need sampling or
  cost-sensitive learning.

## How to run

```bash
pip install -r requirements.txt
python train.py
```

Charts are saved to `images/`.

## Project structure

```
customer-churn-prediction/
├── data_loader.py      # synthetic dataset generator (pandas/numpy)
├── train.py            # train/evaluate LR + RF, save charts
├── images/             # confusion matrices, feature importance
├── requirements.txt
└── README.md
```

## Skills demonstrated

Python · pandas · scikit-learn · matplotlib · data storytelling
