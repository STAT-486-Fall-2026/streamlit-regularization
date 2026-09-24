# Polynomial regularization lab

An interactive Streamlit app for exploring how L1 (LASSO), L2 (Ridge),
elastic-net, and unregularized polynomial regression behave as model degree and
regularization strength change.

The app generates a reproducible polynomial data set, fits a selected model,
and displays:

- Training and test RMSE
- An interactive data, truth, and fitted-curve chart
- The train-test generalization gap
- Active polynomial terms and a coefficient profile
- The fitted equation and penalty definitions

## Run locally

This project uses [uv](https://docs.astral.sh/uv/) for its Python environment
and dependency lockfile.

```powershell
uv sync
uv run streamlit run streamlit_app.py
```

Streamlit will print the local URL, normally <http://localhost:8501>.

## Run tests

```powershell
uv run pytest
```

The tests cover the simulation and fitting logic as well as a headless
Streamlit smoke test.
