"""Interactive demonstration of regularized polynomial regression."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from regression import (
    Regularization,
    fit_polynomial_model,
    generate_polynomial_data,
)

st.set_page_config(
    page_title="Polynomial regularization lab",
    page_icon=":material/show_chart:",
    layout="wide",
)

REGULARIZATION_OPTIONS: list[Regularization] = [
    "None",
    "L1 (LASSO)",
    "L2 (Ridge)",
    "Elastic net",
]
LAMBDA_OPTIONS = [
    0.0001,
    0.0003,
    0.001,
    0.003,
    0.01,
    0.03,
    0.1,
    0.3,
    1.0,
    3.0,
    10.0,
    30.0,
    100.0,
]


def format_lambda(value: float) -> str:
    if value < 0.001:
        return f"{value:.4f}"
    if value < 0.1:
        return f"{value:.3f}".rstrip("0")
    return f"{value:g}"


def human_readable_bounds(values: np.ndarray, target_ticks: int = 6) -> list[float]:
    """Return rounded bounds that produce readable axis tick labels."""
    finite_values = values[np.isfinite(values)]
    minimum = float(finite_values.min())
    maximum = float(finite_values.max())
    span = maximum - minimum

    if span == 0:
        padding = max(abs(minimum) * 0.1, 1.0)
        minimum -= padding
        maximum += padding
        span = maximum - minimum

    rough_step = span / target_ticks
    magnitude = 10 ** np.floor(np.log10(rough_step))
    normalized_step = rough_step / magnitude
    if normalized_step <= 1:
        step = magnitude
    elif normalized_step <= 2:
        step = 2 * magnitude
    elif normalized_step <= 5:
        step = 5 * magnitude
    else:
        step = 10 * magnitude

    return [
        float(np.floor(minimum / step) * step),
        float(np.ceil(maximum / step) * step),
    ]


def format_equation_latex(
    label: str,
    intercept: float,
    coefficients: np.ndarray,
    terms_per_line: int = 4,
) -> str:
    """Format a polynomial as a readable, line-wrapped LaTeX equation."""
    terms = [f"{intercept:.3f}"]
    terms.extend(
        f"{'+' if coefficient >= 0 else '-'} {abs(coefficient):.3f}x^{{{degree}}}"
        for degree, coefficient in enumerate(coefficients, start=1)
        if abs(coefficient) >= 0.0005
    )
    lines = [f"{label} = {terms[0]}"]
    lines.extend(
        "&\\quad " + " ".join(terms[start : start + terms_per_line])
        for start in range(1, len(terms), terms_per_line)
    )
    return r"\begin{aligned}" + r"\\".join(lines) + r"\end{aligned}"


def interpretation(
    true_degree: int,
    model_degree: int,
    regularization: Regularization,
    lambda_value: float,
    active_coefficients: int,
) -> str:
    if model_degree < true_degree:
        return (
            "The fitted model is less flexible than the true process, so it has "
            "structural bias that regularization cannot remove."
        )
    if regularization == "None" and model_degree > true_degree:
        return (
            "The model has extra degrees of freedom and no penalty. Watch for a "
            "small training error paired with a larger test error."
        )
    if regularization == "L1 (LASSO)":
        return (
            f"LASSO has retained {active_coefficients} of {model_degree} polynomial "
            "terms. Increase λ to encourage a sparser model."
        )
    if regularization == "L2 (Ridge)":
        return (
            "Ridge continuously shrinks every coefficient. Larger λ usually makes "
            "the curve smoother, trading variance for bias."
        )
    if regularization == "Elastic net":
        return (
            "Elastic net combines LASSO sparsity with Ridge shrinkage. Its mix is "
            "controlled by the L1 ratio."
        )
    return "The selected model degree matches the true polynomial degree."


st.title("Polynomial regularization lab", icon=":material/show_chart:")
st.caption(
    "Explore how model flexibility and coefficient penalties change fit, "
    "generalization, and sparsity. Drag or scroll on the chart to pan and zoom."
)

with st.sidebar:
    st.header("Experiment controls", icon=":material/tune:")
    true_degree = st.slider(
        "True polynomial degree",
        min_value=1,
        max_value=12,
        value=3,
        key="true_degree",
        help="Degree used to generate the noiseless response.",
    )
    model_degree = st.slider(
        "Fitted model degree",
        min_value=1,
        max_value=20,
        value=10,
        key="model_degree",
        help="Number of polynomial terms available to the fitted model.",
    )
    regularization = st.segmented_control(
        "Regularization",
        REGULARIZATION_OPTIONS,
        default="L2 (Ridge)",
        required=True,
        width="stretch",
        key="regularization",
    )
    lambda_value = st.select_slider(
        "Regularization strength λ",
        options=LAMBDA_OPTIONS,
        value=0.1,
        format_func=format_lambda,
        disabled=regularization == "None",
        key="lambda_value",
        help="Larger values impose stronger coefficient shrinkage.",
    )
    l1_ratio = st.slider(
        "L1 ratio",
        min_value=0.05,
        max_value=0.95,
        value=0.5,
        step=0.05,
        disabled=regularization != "Elastic net",
        key="l1_ratio",
        help="0 is Ridge-like; 1 is LASSO-like.",
    )

    with st.expander("Data settings", icon=":material/database:"):
        n_train = st.slider(
            "Training observations", 15, 200, 40, 5, key="n_train"
        )
        n_test = st.slider("Test observations", 20, 500, 200, 20, key="n_test")
        noise_std = st.slider(
            "Noise standard deviation", 0.0, 1.0, 0.2, 0.05, key="noise_std"
        )
        seed = st.number_input(
            "Random seed", min_value=0, max_value=100_000, value=486, key="seed"
        )
    st.caption("Change the seed for a new reproducible sample and true polynomial.")

data = generate_polynomial_data(
    true_degree=true_degree,
    n_train=n_train,
    n_test=n_test,
    noise_std=noise_std,
    seed=int(seed),
)
curve_x = np.linspace(-1.0, 1.0, 500)
result = fit_polynomial_model(
    data=data,
    model_degree=model_degree,
    regularization=regularization,
    lambda_value=lambda_value,
    l1_ratio=l1_ratio,
    curve_x=curve_x,
)

generalization_gap = result.test_rmse - result.train_rmse
with st.container(horizontal=True):
    st.metric("Training RMSE", f"{result.train_rmse:.3f}", border=True)
    st.metric("Test RMSE", f"{result.test_rmse:.3f}", border=True)
    st.metric(
        "Generalization gap",
        f"{generalization_gap:+.3f}",
        help="Test RMSE minus training RMSE.",
        border=True,
    )
    st.metric(
        "Active terms",
        f"{result.active_coefficients} / {model_degree}",
        help="Coefficients with absolute value greater than 10⁻⁸.",
        border=True,
    )

point_frame = pd.concat(
    [
        pd.DataFrame({"x": data.x_train, "y": data.y_train, "sample": "Training"}),
        pd.DataFrame({"x": data.x_test, "y": data.y_test, "sample": "Test"}),
    ],
    ignore_index=True,
)
true_curve = pd.DataFrame(
    {"x": curve_x, "y": data.true_values(curve_x), "curve": "True function"}
)
fitted_curve = pd.DataFrame(
    {"x": curve_x, "y": result.curve_predictions, "curve": "Fitted model"}
)
y_domain = human_readable_bounds(
    np.concatenate(
        [
            point_frame["y"].to_numpy(),
            true_curve["y"].to_numpy(),
            fitted_curve["y"].to_numpy(),
        ]
    )
)

points = (
    alt.Chart(point_frame)
    .mark_circle(size=62, opacity=0.72)
    .encode(
        x=alt.X(
            "x:Q",
            title="Predictor x",
            scale=alt.Scale(domain=[-1, 1], nice=False),
        ),
        y=alt.Y(
            "y:Q",
            title="Response y",
            scale=alt.Scale(domain=y_domain, nice=False),
        ),
        color=alt.Color(
            "sample:N",
            title="Observations",
            scale=alt.Scale(
                domain=["Training", "Test"], range=["#E45756", "#54A24B"]
            ),
        ),
        tooltip=[
            alt.Tooltip("sample:N", title="Sample"),
            alt.Tooltip("x:Q", format=".3f"),
            alt.Tooltip("y:Q", format=".3f"),
        ],
    )
)
truth_line = (
    alt.Chart(true_curve)
    .mark_line(color="#6B7280", strokeWidth=3, strokeDash=[7, 5])
    .encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[-1, 1], nice=False)),
        y=alt.Y("y:Q", scale=alt.Scale(domain=y_domain, nice=False)),
        tooltip=[
            alt.Tooltip("curve:N", title="Curve"),
            alt.Tooltip("x:Q", format=".3f"),
            alt.Tooltip("y:Q", format=".3f"),
        ],
    )
)
fit_line = (
    alt.Chart(fitted_curve)
    .mark_line(color="#2563EB", strokeWidth=4)
    .encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[-1, 1], nice=False)),
        y=alt.Y("y:Q", scale=alt.Scale(domain=y_domain, nice=False)),
        tooltip=[
            alt.Tooltip("curve:N", title="Curve"),
            alt.Tooltip("x:Q", format=".3f"),
            alt.Tooltip("y:Q", format=".3f"),
        ],
    )
)

chart_col, equation_col = st.columns([2, 1], gap="large", vertical_alignment="top")
with chart_col:
    st.subheader("Observed data and fitted curve", icon=":material/monitoring:")
    st.caption("Blue: fitted model · dashed gray: true function · red/green: observations")
    fit_chart = (
        (truth_line + fit_line + points)
        .properties(width=600, height=600)
        .configure_axis(gridColor="#E5E7EB", gridOpacity=0.45)
        .interactive()
    )
    st.altair_chart(fit_chart, width="content", key="fit_chart")

with equation_col:
    with st.container(border=True, height=600):
        st.subheader("Model equations", icon=":material/functions:")
        st.caption("Coefficients are rounded to three decimals; tiny terms are omitted.")
        st.markdown("**True underlying polynomial**")
        st.latex(
            format_equation_latex(
                "y",
                float(data.true_coefficients[0]),
                data.true_coefficients[1:],
            )
        )
        st.markdown("**Fitted polynomial**")
        st.latex(
            format_equation_latex(r"\hat{y}", result.intercept, result.raw_coefficients)
        )

coefficient_frame = pd.DataFrame(
    {
        "term": [f"x^{degree}" for degree in range(1, model_degree + 1)],
        "degree": range(1, model_degree + 1),
        "coefficient": result.raw_coefficients,
        "sign": np.where(result.raw_coefficients >= 0, "Positive", "Negative"),
    }
)
coefficient_chart = (
    alt.Chart(coefficient_frame)
    .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
    .encode(
        x=alt.X("term:N", title="Polynomial term", sort=alt.SortField("degree")),
        y=alt.Y("coefficient:Q", title="Coefficient"),
        color=alt.Color(
            "sign:N",
            title="Sign",
            scale=alt.Scale(
                domain=["Positive", "Negative"], range=["#2563EB", "#E45756"]
            ),
        ),
        tooltip=[
            alt.Tooltip("term:N", title="Term"),
            alt.Tooltip("coefficient:Q", format=".5f"),
        ],
    )
    .properties(height=285)
)

details_col, coefficient_col = st.columns([1, 2])
with details_col:
    with st.container(border=True, height="stretch"):
        st.subheader("What to notice", icon=":material/lightbulb:")
        st.write(
            interpretation(
                true_degree,
                model_degree,
                regularization,
                lambda_value,
                result.active_coefficients,
            )
        )
        st.metric("Coefficient L1 norm", f"{result.coefficient_l1_norm:.3f}")
        if regularization != "None":
            st.caption(
                f"Current penalty: {regularization}, λ = {format_lambda(lambda_value)}"
            )
        else:
            st.caption("Ordinary least squares: no coefficient penalty.")

with coefficient_col:
    with st.container(border=True, height="stretch"):
        st.subheader("Coefficient profile", icon=":material/bar_chart:")
        st.caption(
            "Coefficients are converted back to the original x-power basis after fitting."
        )
        st.altair_chart(coefficient_chart, key="coefficient_chart")

with st.expander("Penalty definitions and model details", icon=":material/function:"):
    st.markdown(
        r"All polynomial features are standardized before fitting. The common loss scale is "
        r"$\frac{1}{2n}\sum_i(y_i-\hat y_i)^2$. LASSO adds "
        r"$\lambda\sum_j|\beta_j|$; Ridge adds "
        r"$\frac{\lambda}{2}\sum_j\beta_j^2$; elastic net adds both, weighted by "
        r"the L1 ratio $\rho$. The intercept is not penalized."
    )
    st.caption(
        "Because high powers of x are strongly correlated, individual raw-basis "
        "coefficients can be large even when the fitted curve is smooth."
    )
