import pytest

from dashboard.lib.config import load_config


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "not-a-number"])
def test_dashboard_timeout_must_be_positive_and_finite(monkeypatch, value):
    monkeypatch.setenv("VERIMIND_DASHBOARD_TIMEOUT_SECONDS", value)

    with pytest.raises(ValueError, match="VERIMIND_DASHBOARD_TIMEOUT_SECONDS"):
        load_config()


@pytest.mark.parametrize("value", ["0", "-1", "not-an-integer"])
def test_dashboard_batch_limit_must_be_positive_integer(monkeypatch, value):
    monkeypatch.setenv("VERIMIND_DASHBOARD_BATCH_LIMIT", value)

    with pytest.raises(ValueError, match="VERIMIND_DASHBOARD_BATCH_LIMIT"):
        load_config()
