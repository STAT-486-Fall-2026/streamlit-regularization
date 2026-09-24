from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).parents[1] / "streamlit_app.py"


def test_app_renders_without_exceptions() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()

    assert not app.exception
    assert app.title[0].value == "Polynomial regularization lab"
    assert len(app.metric) == 5


def test_switching_to_lasso_reruns_cleanly() -> None:
    app = AppTest.from_file(str(APP_PATH), default_timeout=15).run()
    app.segmented_control(key="regularization").set_value("L1 (LASSO)").run()

    assert not app.exception
    assert app.metric[3].value.endswith("/ 10")
