# quant-research-system

quant-research-system is a research-oriented quantitative trading project focused on building and evaluating an equity long-only strategy pipeline.

The repository is designed to demonstrate end-to-end quant engineering capabilities, including:

- data preparation and universe construction
- feature engineering workflows
- alpha model training and evaluation
- portfolio backtesting and risk analysis
- strategy parameter sweeps and reporting

## Project Goal

The goal of this project is to provide a modular framework for turning market data into tradable portfolio decisions, then validating performance through reproducible backtests and diagnostics.

## What This Repository Highlights

- **System design**: clear separation between data, features, modeling, and backtesting modules  
- **Research workflow**: iterative experiments for strategy tuning and risk/return trade-offs  
- **Engineering practices**: script-based pipelines, structured outputs, and test coverage  

## Repository Structure

- `core/data_build`: data ingestion and canonicalization
- `core/universe`: tradable universe construction rules
- `core/features`: feature pipeline components
- `core/models/Xgboost`: alpha modeling, prediction, and evaluation
- `core/backtest`: strategy simulation, parameter sweeps, and reports
- `dq`: data quality checks
- `tests`: validation and guardrail tests

## Notes on Public Version

This public repository is intentionally sanitized for portfolio and interview purposes.
Certain strategy-sensitive details (for example, proprietary alpha definitions and some production configurations) are omitted or abstracted.

The focus of this version is to showcase the architecture, workflow, and implementation quality of a practical quant research system.
