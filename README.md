# Generalized Ensemble Optimization

Archival code and dissertation for my 2019 MSc Computational Finance project at the University of Essex.

The project proposes a **Generalized Ensemble Optimizer (GEO)**: a framework for selecting model hyperparameters, blending model predictions, and then using a genetic algorithm to search for additional models that improve the ensemble.

The original implementation used:

- Logistic regression
- Stochastic gradient descent classification
- Support vector classification
- XGBoost
- Bayesian hyperparameter optimization via Hyperopt
- Genetic search over candidate ensemble additions
- Numerai competition data

The dissertation PDF is included in [`docs/Generalized-Ensemble-Optimization.pdf`](docs/Generalized-Ensemble-Optimization.pdf).

## Repository Status

This is **historical research code**, preserved close to the original 2019 project listing. It is not a modernized package and should not be treated as production software.

Known limitations:

- The original Numerai CSV data files are not committed because they are large external data artifacts.
- The code expects local CSV filenames from the original project.
- Dependency APIs may have changed since 2019, especially `scikit-learn` and `xgboost`.
- The evaluation metric is named `AUC` in parts of the code, but the implemented calculation is rounded binary accuracy.

## Project Structure

```text
.
├── docs/
│   └── Generalized-Ensemble-Optimization.pdf
├── src/
│   ├── Bayes.py       # Bayesian optimization and model blending
│   ├── GA.py          # Genetic search over ensemble additions
│   ├── Load_Data.py   # Numerai CSV loading helpers
│   └── ML.py          # Model and hyperparameter-space definitions
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Data

The original project used Numerai CSV files with these expected local names:

```text
numerai_training_data.csv
numerai_tournament_data.csv
```

Place those files in the working directory before running the scripts. They are intentionally excluded from Git via `.gitignore`.

## Install

Use a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For best reproducibility, use older package versions from the 2019 Python ecosystem. The current `requirements.txt` intentionally lists packages without pinned versions because the original environment lockfile is not available.

## Running

From the repository root, run:

```bash
cd src
python Bayes.py
```

`Bayes.py` loads the Numerai data, optimizes individual model hyperparameters, blends model outputs, and writes a local pickle artifact.

Then:

```bash
python GA.py
```

`GA.py` loads the prior blend artifact and searches for additional candidate models that improve the ensemble.

## Research Summary

The GEO workflow has three stages:

1. Optimize each base learner's hyperparameters with Bayesian optimization.
2. Blend the optimized base learners into a single ensemble predictor.
3. Use genetic search to propose additional models and retain candidates with positive marginal contribution to the ensemble.

The motivation was to automate part of the model-selection process for machine learning problems where the useful ensemble is not known in advance.

## Citation

```text
Overing, Matthew. Generalized Ensemble Optimization. MSc Computational Finance dissertation, University of Essex, 2019.
```

