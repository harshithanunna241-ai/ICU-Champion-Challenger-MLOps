"""Unit tests for Champion-Challenger promotion audit logging."""

import json
from datetime import datetime, timezone

import pytest

from src.evaluation.promotion_audit import (
    save_promotion_audit,
    read_promotion_audit,
)


@pytest.fixture
def promotion_result():
    return {
        "decision": "PROMOTE",
        "promotion_approved": True,
        "current_champion": "Logistic Regression",
        "challenger": "XGBoost",
        "reasons": [
            "Challenger achieved higher AUPRC.",
            "Challenger passed promotion criteria.",
        ],
    }


@pytest.fixture
def champion_metrics():
    return {
        "auprc": 0.82,
        "auroc": 0.88,
        "f1_score": 0.79,
    }


@pytest.fixture
def challenger_metrics():
    return {
        "auprc": 0.89,
        "auroc": 0.92,
        "f1_score": 0.85,
    }


# ---------------------------------------------------------
# Tests for save_promotion_audit
# ---------------------------------------------------------


def test_save_returns_dictionary(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    result = save_promotion_audit(
        promotion_result,
        champion_metrics,
        challenger_metrics,
        audit_dir=tmp_path,
    )

    assert isinstance(result, dict)


def test_save_returns_expected_keys(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    result = save_promotion_audit(
        promotion_result,
        champion_metrics,
        challenger_metrics,
        audit_dir=tmp_path,
    )

    assert set(result.keys()) == {
        "audit_file",
        "audit_record",
    }


def test_creates_audit_directory(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    audit_dir = tmp_path / "nested" / "audit"

    save_promotion_audit(
        promotion_result,
        champion_metrics,
        challenger_metrics,
        audit_dir=audit_dir,
    )

    assert audit_dir.exists()
    assert audit_dir.is_dir()


def test_creates_jsonl_file(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    result = save_promotion_audit(
        promotion_result,
        champion_metrics,
        challenger_metrics,
        audit_dir=tmp_path,
    )

    audit_file = tmp_path / "promotion_audit.jsonl"

    assert audit_file.exists()
    assert result["audit_file"] == str(audit_file)


def test_returns_expected_audit_record(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    result = save_promotion_audit(
        promotion_result,
        champion_metrics,
        challenger_metrics,
        audit_dir=tmp_path,
    )

    record = result["audit_record"]

    assert record["decision"] == "PROMOTE"
    assert record["promotion_approved"] is True
    assert record["current_champion"] == "Logistic Regression"
    assert record["challenger"] == "XGBoost"
    assert record["champion_metrics"] == champion_metrics
    assert record["challenger_metrics"] == challenger_metrics
    assert record["reasons"] == promotion_result["reasons"]


def test_audit_record_contains_timestamp(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    result = save_promotion_audit(
        promotion_result,
        champion_metrics,
        challenger_metrics,
        audit_dir=tmp_path,
    )

    timestamp = result["audit_record"]["timestamp"]

    parsed_timestamp = datetime.fromisoformat(timestamp)

    assert parsed_timestamp.tzinfo is not None
    assert parsed_timestamp.utcoffset().total_seconds() == 0


def test_timestamp_uses_utc_timezone(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    result = save_promotion_audit(
        promotion_result,
        champion_metrics,
        challenger_metrics,
        audit_dir=tmp_path,
    )

    timestamp = result["audit_record"]["timestamp"]

    parsed_timestamp = datetime.fromisoformat(timestamp)

    assert parsed_timestamp.utcoffset() == timezone.utc.utcoffset(
        parsed_timestamp
    )


def test_appends_newline_after_record(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    result = save_promotion_audit(
        promotion_result,
        champion_metrics,
        challenger_metrics,
        audit_dir=tmp_path,
    )

    audit_file = tmp_path / "promotion_audit.jsonl"

    content = audit_file.read_text(encoding="utf-8")

    assert content.endswith("\n")
    assert result["audit_record"]["decision"] in content


def test_saved_record_is_valid_json(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    save_promotion_audit(
        promotion_result,
        champion_metrics,
        challenger_metrics,
        audit_dir=tmp_path,
    )

    audit_file = tmp_path / "promotion_audit.jsonl"

    record = json.loads(
        audit_file.read_text(encoding="utf-8").strip()
    )

    assert record["decision"] == "PROMOTE"


def test_saves_multiple_records(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    save_promotion_audit(
        promotion_result,
        champion_metrics,
        challenger_metrics,
        audit_dir=tmp_path,
    )

    second_result = promotion_result.copy()
    second_result["decision"] = "REJECT"
    second_result["promotion_approved"] = False

    save_promotion_audit(
        second_result,
        champion_metrics,
        challenger_metrics,
        audit_dir=tmp_path,
    )

    audit_file = tmp_path / "promotion_audit.jsonl"

    lines = audit_file.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert json.loads(lines[0])["decision"] == "PROMOTE"
    assert json.loads(lines[1])["decision"] == "REJECT"


def test_rejected_promotion_is_saved(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    promotion_result["decision"] = "REJECT"
    promotion_result["promotion_approved"] = False

    result = save_promotion_audit(
        promotion_result,
        champion_metrics,
        challenger_metrics,
        audit_dir=tmp_path,
    )

    record = result["audit_record"]

    assert record["decision"] == "REJECT"
    assert record["promotion_approved"] is False


def test_accepts_empty_metrics_dictionaries(
    tmp_path,
    promotion_result,
):
    result = save_promotion_audit(
        promotion_result,
        {},
        {},
        audit_dir=tmp_path,
    )

    assert result["audit_record"]["champion_metrics"] == {}
    assert result["audit_record"]["challenger_metrics"] == {}


def test_accepts_empty_reasons_list(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    promotion_result["reasons"] = []

    result = save_promotion_audit(
        promotion_result,
        champion_metrics,
        challenger_metrics,
        audit_dir=tmp_path,
    )

    assert result["audit_record"]["reasons"] == []


def test_preserves_nested_metric_values(
    tmp_path,
    promotion_result,
):
    champion = {
        "metrics": {
            "auprc": 0.82,
            "details": {
                "precision": 0.9,
                "recall": 0.75,
            },
        }
    }

    challenger = {
        "metrics": {
            "auprc": 0.89,
            "details": {
                "precision": 0.92,
                "recall": 0.81,
            },
        }
    }

    result = save_promotion_audit(
        promotion_result,
        champion,
        challenger,
        audit_dir=tmp_path,
    )

    assert result["audit_record"]["champion_metrics"] == champion
    assert result["audit_record"]["challenger_metrics"] == challenger


# ---------------------------------------------------------
# Input validation tests for save_promotion_audit
# ---------------------------------------------------------


@pytest.mark.parametrize(
    "promotion_result",
    [
        None,
        [],
        "PROMOTE",
        123,
    ],
)
def test_rejects_non_dictionary_promotion_result(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    with pytest.raises(
        TypeError,
        match="promotion_result must be a dictionary",
    ):
        save_promotion_audit(
            promotion_result,
            champion_metrics,
            challenger_metrics,
            audit_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "champion_metrics",
    [
        None,
        [],
        "metrics",
        123,
    ],
)
def test_rejects_non_dictionary_champion_metrics(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    with pytest.raises(
        TypeError,
        match="champion_metrics must be a dictionary",
    ):
        save_promotion_audit(
            promotion_result,
            champion_metrics,
            challenger_metrics,
            audit_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "challenger_metrics",
    [
        None,
        [],
        "metrics",
        123,
    ],
)
def test_rejects_non_dictionary_challenger_metrics(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    with pytest.raises(
        TypeError,
        match="challenger_metrics must be a dictionary",
    ):
        save_promotion_audit(
            promotion_result,
            champion_metrics,
            challenger_metrics,
            audit_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "missing_field",
    [
        "decision",
        "promotion_approved",
        "current_champion",
        "challenger",
        "reasons",
    ],
)
def test_rejects_missing_required_fields(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
    missing_field,
):
    invalid_result = promotion_result.copy()
    invalid_result.pop(missing_field)

    with pytest.raises(
        ValueError,
        match="Missing promotion result fields",
    ):
        save_promotion_audit(
            invalid_result,
            champion_metrics,
            challenger_metrics,
            audit_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "reasons",
    [
        "reason",
        None,
        {},
        123,
    ],
)
def test_rejects_non_list_reasons(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
    reasons,
):
    invalid_result = promotion_result.copy()
    invalid_result["reasons"] = reasons

    with pytest.raises(
        TypeError,
        match="Promotion reasons must be a list",
    ):
        save_promotion_audit(
            invalid_result,
            champion_metrics,
            challenger_metrics,
            audit_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "approved",
    [
        "True",
        1,
        0,
        None,
        [],
    ],
)
def test_rejects_non_boolean_promotion_approved(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
    approved,
):
    invalid_result = promotion_result.copy()
    invalid_result["promotion_approved"] = approved

    with pytest.raises(
        TypeError,
        match="promotion_approved must be a boolean",
    ):
        save_promotion_audit(
            invalid_result,
            champion_metrics,
            challenger_metrics,
            audit_dir=tmp_path,
        )


def test_rejects_nan_values_in_metrics(
    tmp_path,
    promotion_result,
    challenger_metrics,
):
    champion = {"auprc": float("nan")}

    with pytest.raises(ValueError):
        save_promotion_audit(
            promotion_result,
            champion,
            challenger_metrics,
            audit_dir=tmp_path,
        )


def test_rejects_infinity_values_in_metrics(
    tmp_path,
    promotion_result,
    challenger_metrics,
):
    champion = {"auprc": float("inf")}

    with pytest.raises(ValueError):
        save_promotion_audit(
            promotion_result,
            champion,
            challenger_metrics,
            audit_dir=tmp_path,
        )


# ---------------------------------------------------------
# Tests for read_promotion_audit
# ---------------------------------------------------------


def test_read_returns_empty_list_when_file_missing(tmp_path):
    result = read_promotion_audit(tmp_path)

    assert result == []


def test_read_returns_list(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    save_promotion_audit(
        promotion_result,
        champion_metrics,
        challenger_metrics,
        audit_dir=tmp_path,
    )

    result = read_promotion_audit(tmp_path)

    assert isinstance(result, list)


def test_read_returns_saved_record(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    save_promotion_audit(
        promotion_result,
        champion_metrics,
        challenger_metrics,
        audit_dir=tmp_path,
    )

    records = read_promotion_audit(tmp_path)

    assert len(records) == 1
    assert records[0]["decision"] == "PROMOTE"
    assert records[0]["challenger"] == "XGBoost"


def test_read_returns_all_saved_records(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    for decision, approved in [
        ("PROMOTE", True),
        ("REJECT", False),
        ("PROMOTE", True),
    ]:
        result = promotion_result.copy()
        result["decision"] = decision
        result["promotion_approved"] = approved

        save_promotion_audit(
            result,
            champion_metrics,
            challenger_metrics,
            audit_dir=tmp_path,
        )

    records = read_promotion_audit(tmp_path)

    assert len(records) == 3


def test_read_preserves_record_order(
    tmp_path,
    promotion_result,
    champion_metrics,
    challenger_metrics,
):
    decisions = ["PROMOTE", "REJECT", "PROMOTE"]

    for decision in decisions:
        result = promotion_result.copy()
        result["decision"] = decision

        save_promotion_audit(
            result,
            champion_metrics,
            challenger_metrics,
            audit_dir=tmp_path,
        )

    records = read_promotion_audit(tmp_path)

    assert [record["decision"] for record in records] == decisions


def test_read_skips_blank_lines(tmp_path):
    audit_file = tmp_path / "promotion_audit.jsonl"

    valid_record = {
        "decision": "PROMOTE",
        "promotion_approved": True,
    }

    audit_file.write_text(
        json.dumps(valid_record) + "\n\n   \n",
        encoding="utf-8",
    )

    records = read_promotion_audit(tmp_path)

    assert records == [valid_record]


def test_read_raises_error_for_invalid_json(tmp_path):
    audit_file = tmp_path / "promotion_audit.jsonl"

    audit_file.write_text(
        '{"decision": "PROMOTE"}\ninvalid-json\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid audit record at line 2",
    ):
        read_promotion_audit(tmp_path)


def test_read_reports_correct_line_number(tmp_path):
    audit_file = tmp_path / "promotion_audit.jsonl"

    audit_file.write_text(
        '{"decision": "PROMOTE"}\n\ninvalid-json\n',
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Invalid audit record at line 3",
    ):
        read_promotion_audit(tmp_path)


def test_read_preserves_nested_json_values(tmp_path):
    audit_file = tmp_path / "promotion_audit.jsonl"

    record = {
        "decision": "PROMOTE",
        "metrics": {
            "auprc": 0.91,
            "details": [0.8, 0.9],
        },
    }

    audit_file.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    records = read_promotion_audit(tmp_path)

    assert records[0] == record


def test_read_uses_jsonl_file_inside_audit_directory(tmp_path):
    audit_file = tmp_path / "promotion_audit.jsonl"

    record = {"decision": "REJECT"}

    audit_file.write_text(
        json.dumps(record) + "\n",
        encoding="utf-8",
    )

    records = read_promotion_audit(tmp_path)

    assert records == [record]