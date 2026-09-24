import numpy as np

from regression import fit_polynomial_model, format_polynomial, generate_polynomial_data


def test_data_generation_is_reproducible_and_has_requested_degree() -> None:
    first = generate_polynomial_data(5, 30, 40, 0.2, 486)
    second = generate_polynomial_data(5, 30, 40, 0.2, 486)

    np.testing.assert_allclose(first.x_train, second.x_train)
    np.testing.assert_allclose(first.y_test, second.y_test)
    assert len(first.true_coefficients) == 6
    assert abs(first.true_coefficients[-1]) >= 0.8


def test_fit_returns_finite_errors_and_expected_coefficients() -> None:
    data = generate_polynomial_data(3, 50, 80, 0.15, 12)
    curve_x = np.linspace(-1.0, 1.0, 101)
    result = fit_polynomial_model(data, 8, "L2 (Ridge)", 0.1, 0.5, curve_x)

    assert result.raw_coefficients.shape == (8,)
    assert result.curve_predictions.shape == curve_x.shape
    assert np.isfinite(result.train_rmse)
    assert np.isfinite(result.test_rmse)


def test_strong_lasso_is_no_less_sparse_than_weak_lasso() -> None:
    data = generate_polynomial_data(3, 60, 100, 0.2, 99)
    curve_x = np.linspace(-1.0, 1.0, 101)
    weak = fit_polynomial_model(data, 12, "L1 (LASSO)", 0.0001, 0.5, curve_x)
    strong = fit_polynomial_model(data, 12, "L1 (LASSO)", 1.0, 0.5, curve_x)

    assert strong.active_coefficients <= weak.active_coefficients
    assert strong.coefficient_l1_norm <= weak.coefficient_l1_norm


def test_equation_formatter_includes_intercept_and_all_terms() -> None:
    equation = format_polynomial(1.0, np.array([2.0, -3.0]))

    assert equation == "ŷ = 1.000 + 2.000x^1 − 3.000x^2"
