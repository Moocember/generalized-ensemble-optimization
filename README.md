# Generalized Ensemble Optimization

A unified loop for AutoML and ensemble construction: tune, blend, and *grow* the ensemble until it stops improving. MSc Computational Finance dissertation, University of Essex, 2019.

## The idea

Most AutoML systems tune hyperparameters for a fixed set of models. Most ensemble methods blend a fixed set of trained models. The **Generalized Ensemble Optimizer (GEO)** closes the loop: it tunes, blends, *and discovers new models that improve the ensemble*, in one procedure.

Three stages, run in sequence:

1. **Tune** — Bayesian optimization (Hyperopt, expected improvement) finds the best hyperparameters for each base learner independently.
2. **Blend** — The tuned learners' predictions are combined into a single meta-model that is more predictive than any individual learner.
3. **Grow** — A genetic algorithm searches the space of *additional* candidate models, retaining only those whose marginal contribution improves the blended ensemble.

The framework is dataset-agnostic and learner-agnostic. The reference implementation uses logistic regression, SGD classification, SVM, and XGBoost on Numerai competition data.

## Why portfolio theory

The motivating insight is that a machine-learning ensemble is a portfolio of biased estimators, and the same math that governs portfolio diversification governs ensemble construction. Ray Dalio's "Holy Grail of Investing" (uncorrelated bets reduce portfolio variance asymptotically) and the Kelly Criterion (more uncorrelated edges raise optimal exposure) translate directly: each base learner is a biased coin, and adding uncorrelated learners lowers collective bias without raising variance. Five truly uncorrelated 60%-accurate predictors in consensus reach 99% accuracy in theory; the engineering challenge is finding learners with low enough cross-correlation to approach that bound. GEO's third stage is exactly that search.

## Result

Tested on Numerai tournament data against an equal-intensity random-search baseline (same models, same dataset, randomly sampled hyperparameters), GEO outperformed the benchmark on out-of-sample classification accuracy. See the [dissertation PDF](docs/Generalized-Ensemble-Optimization.pdf), Results section, for the full evaluation.

## Repository status

Historical research code, preserved close to the original 2019 listing. Not a modernized package; not production software. Known caveats: `scikit-learn` and `xgboost` APIs have shifted since 2019; the metric named `AUC` in the code is implemented as rounded binary accuracy.

## Project structure

```text
.
├── docs/
│   └── Generalized-Ensemble-Optimization.pdf
├── src/
│   ├── Bayes.py       # Bayesian hyperparameter optimization and blending
│   ├── GA.py          # Genetic search over candidate ensemble additions
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

Place them in the working directory before running. Excluded from Git via `.gitignore`.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Older 2019 package versions give the most faithful reproduction. The original lockfile is not available; `requirements.txt` is unpinned.

## Running

```bash
cd src
python Bayes.py   # tune base learners, blend predictions, write pickle artifact
python GA.py      # genetic search for additional models that improve the blend
```

## Citation

```text
Overing, Matthew. Generalized Ensemble Optimization.
MSc Computational Finance dissertation, University of Essex, 2019.
```
