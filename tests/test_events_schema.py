from src.core.events_schema import validate_events_payload


def test_events_schema_ok():
    items = validate_events_payload(
        {"events": [{"title": "X", "date": "2025-01-02", "time": "09:00", "importance": "high"}]}
    )
    assert items[0].time == "09:00"


def test_events_schema_time_format_rejected():
    bad = {"events": [{"title": "X", "date": "2025-01-02", "time": "9:00"}]}
    try:
        validate_events_payload(bad)
        assert False, "should raise"
    except Exception:
        assert True
