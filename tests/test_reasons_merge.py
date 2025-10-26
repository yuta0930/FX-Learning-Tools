import pandas as pd
from src.policy.reasons import unify_reasons_df


def test_unify_reasons_basic():
    df = pd.DataFrame(
        {
            "gate_reason": ["ruleA", "", None, "ruleB"],
            "deny_reason": ["env", "spread>1.0", "", None],
        }
    )
    out = unify_reasons_df(df)
    assert "reason" in out.columns
    assert out.loc[0, "reason"] == "ruleA | env"
    assert out.loc[1, "reason"] == "spread>1.0"
    assert out.loc[2, "reason"] == ""
    assert out.loc[3, "reason"] == "ruleB"
