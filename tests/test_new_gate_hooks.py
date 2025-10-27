from components import gate_decision


def test_gate_decision_basic():
    row = {
        "p_cal": 0.9,
        "quality": 0.9,
        "session": "tokyo",
        "atr_15m": 0.2,
        "spread": 0.1,
        "news_flag": False,
        "variant": "A",
    }
    ok, meta = gate_decision(row)
    assert ok is True
