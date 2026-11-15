# Project Overview

## Problem
Digital payment networks contain interacting accounts and transactions. Looking at each payment independently can hide patterns involving repeated counterparties, unusually connected accounts, and clusters of activity.

## Proposed approach
GraphShield represents accounts as nodes and transfers as directed edges. Transaction attributes and account-level network statistics are combined to estimate a fraud probability for each transaction. A compact GraphSAGE-style message-passing classifier is compared with a Random Forest baseline.

## Analytics
- Descriptive: transaction counts, amounts, channels and fraud-label distribution.
- Network: node/edge counts, in/out degree and account activity.
- Predictive: holdout fraud classification metrics and transaction risk bands.
- Intelligence: ranked transactions and account relationship exploration.

## Evaluation
Accuracy, precision, recall, F1 and ROC-AUC are reported on a stratified holdout split. Because the sample data is synthetic, these metrics are software-demo diagnostics only and must not be interpreted as operational performance.
