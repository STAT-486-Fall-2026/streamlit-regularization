"""Core simulation and polynomial regression logic for the Streamlit app."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.base import RegressorMixin
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

Regularization = Literal["None", "L1 (LASSO)", "L2 (Ridge)", "Elastic net"]


@dataclass(frozen=True)
class PolynomialData:
    """A reproducible train/test sample from a polynomial data-generating process."""

    x_train: NDArray[np.float64]
    y_train: NDArray[np.float64]
    x_test: NDArray[np.float64]
    y_test: NDArray[np.float64]
    true_coefficients: NDArray[np.float64]

    def true_values(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.polynomial.polynomial.polyval(x, self.true_coefficients)


@dataclass(frozen=True)
class FitResult:
    """Predictions, errors, and coefficients from one fitted model."""

    train_predictions: NDArray[np.float64]
    test_predictions: NDArray[np.float64]
    curve_predictions: NDArray[np.float64]
    raw_coefficients: NDArray[np.float64]
    intercept: float
    train_rmse: float
    test_rmse: float

    @property
    def active_coefficients(self) -> int:
        return int(np.count_nonzero(np.abs(self.raw_coefficients) > 1e-8))

    @property
    def coefficient_l1_norm(self) -> float:
        return float(np.abs(self.raw_coefficients).sum())


def generate_polynomial_data(
    true_degree: int,
    n_train: int,
    n_test: int,
    noise_std: float,
    seed: int,
) -> PolynomialData:
    """Generate independent training and test samples on [-1, 1]."""
    if true_degree < 1:
        raise ValueError("true_degree must be at least 1")
    if n_train < 2 or n_test < 2:
        raise ValueError("n_train and n_test must both be at least 2")
    if noise_std < 0:
        raise ValueError("noise_std cannot be negative")

    rng = np.random.default_rng(seed)
    degrees = np.arange(true_degree + 1)
    # Give higher-order terms more influence so the generated curves visibly wiggle.
    coefficient_scales = 2.0 / np.sqrt(degrees + 1)
    coefficients = rng.normal(0.0, coefficient_scales).astype(float)

    # Keep the requested degree genuinely present and visually consequential.
    leading_sign = 1.0 if coefficients[-1] >= 0 else -1.0
    coefficients[-1] = leading_sign * max(abs(coefficients[-1]), 1.2)

    x_train = rng.uniform(-1.0, 1.0, n_train)
    x_test = rng.uniform(-1.0, 1.0, n_test)
    train_signal = np.polynomial.polynomial.polyval(x_train, coefficients)
    test_signal = np.polynomial.polynomial.polyval(x_test, coefficients)
    y_train = train_signal + rng.normal(0.0, noise_std, n_train)
    y_test = test_signal + rng.normal(0.0, noise_std, n_test)

    return PolynomialData(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        true_coefficients=coefficients,
    )


def _make_estimator(
    regularization: Regularization,
    lambda_value: float,
    l1_ratio: float,
    n_train: int,
) -> RegressorMixin:
    """Create an estimator using a common normalized objective for lambda."""
    if regularization == "None":
        return LinearRegression()
    if lambda_value <= 0:
        raise ValueError("lambda_value must be positive for regularized models")
    if regularization == "L1 (LASSO)":
        return Lasso(alpha=lambda_value, max_iter=100_000, tol=1e-7)
    if regularization == "L2 (Ridge)":
        # Ridge minimizes RSS + alpha * ||beta||^2. Scaling alpha by n makes
        # lambda match (RSS / 2n) + (lambda / 2) * ||beta||^2.
        return Ridge(alpha=n_train * lambda_value)
    if regularization == "Elastic net":
        return ElasticNet(
            alpha=lambda_value,
            l1_ratio=l1_ratio,
            max_iter=100_000,
            tol=1e-7,
        )
    raise ValueError(f"Unknown regularization type: {regularization}")


def fit_polynomial_model(
    data: PolynomialData,
    model_degree: int,
    regularization: Regularization,
    lambda_value: float,
    l1_ratio: float,
    curve_x: NDArray[np.float64],
) -> FitResult:
    """Fit a standardized polynomial model and return raw-basis coefficients."""
    if model_degree < 1:
        raise ValueError("model_degree must be at least 1")

    polynomial = PolynomialFeatures(degree=model_degree, include_bias=False)
    train_features = polynomial.fit_transform(data.x_train.reshape(-1, 1))
    test_features = polynomial.transform(data.x_test.reshape(-1, 1))
    curve_features = polynomial.transform(curve_x.reshape(-1, 1))

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(train_features)
    scaled_test = scaler.transform(test_features)
    scaled_curve = scaler.transform(curve_features)

    estimator = _make_estimator(
        regularization=regularization,
        lambda_value=lambda_value,
        l1_ratio=l1_ratio,
        n_train=len(data.x_train),
    )
    estimator.fit(scaled_train, data.y_train)

    scaled_coefficients = np.asarray(estimator.coef_, dtype=float)
    raw_coefficients = scaled_coefficients / scaler.scale_
    raw_intercept = float(
        estimator.intercept_
        - np.dot(scaled_coefficients, scaler.mean_ / scaler.scale_)
    )
    train_predictions = np.asarray(estimator.predict(scaled_train), dtype=float)
    test_predictions = np.asarray(estimator.predict(scaled_test), dtype=float)
    curve_predictions = np.asarray(estimator.predict(scaled_curve), dtype=float)

    return FitResult(
        train_predictions=train_predictions,
        test_predictions=test_predictions,
        curve_predictions=curve_predictions,
        raw_coefficients=raw_coefficients,
        intercept=raw_intercept,
        train_rmse=float(root_mean_squared_error(data.y_train, train_predictions)),
        test_rmse=float(root_mean_squared_error(data.y_test, test_predictions)),
    )


def format_polynomial(intercept: float, coefficients: NDArray[np.float64]) -> str:
    """Format a polynomial as a compact human-readable equation."""
    terms = [f"{intercept:.3f}"]
    for degree, coefficient in enumerate(coefficients, start=1):
        sign = "+" if coefficient >= 0 else "−"
        terms.append(f" {sign} {abs(coefficient):.3f}x^{degree}")
    return "ŷ = " + "".join(terms)
