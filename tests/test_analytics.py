"""
tests/test_analytics.py
-------------------------
Tests app/data/analytics.py against the hand-crafted sample_df fixture,
where the expected percentages are known exactly (Region West / Product A
revenue is precisely halved in the current period).
"""
import pytest

from app.data.analytics import analyze_sales, analyze_region, analyze_products, get_kpi_summary


def test_analyze_sales_detects_the_known_drop(sample_df):
    result = analyze_sales(sample_df, period_days=30)

    # Overall revenue: previous = 30*(1000+500)=45000, current = 30*(500+500)=30000
    # -> (30000-45000)/45000 = -33.3%
    assert result["revenue_change_pct"] == pytest.approx(-33.3, abs=0.1)

    # West should be the top (most negative) region contributor.
    top_region = result["top_region_contributors"][0]
    assert top_region["region"] == "West"
    assert top_region["change_pct"] == pytest.approx(-50.0, abs=0.1)

    # East should be unchanged.
    east = next(r for r in result["top_region_contributors"] if r["region"] == "East")
    assert east["change_pct"] == pytest.approx(0.0, abs=0.1)


def test_analyze_region_west(sample_df):
    result = analyze_region(sample_df, "West", period_days=30)
    assert result["revenue_change_pct"] == pytest.approx(-50.0, abs=0.1)
    assert result["top_product_contributors"][0]["product"] == "Product A"


def test_analyze_region_unknown_returns_error(sample_df):
    result = analyze_region(sample_df, "Atlantis")
    assert "error" in result
    assert "available_regions" in result


def test_analyze_region_case_insensitive(sample_df):
    result = analyze_region(sample_df, "west")  # lowercase
    assert "error" not in result
    assert result["region"] == "west"  # echoes input, but data matched


def test_analyze_products_specific_product(sample_df):
    result = analyze_products(sample_df, product="Product A")
    assert result["revenue_change_pct"] == pytest.approx(-50.0, abs=0.1)


def test_analyze_products_ranking_all(sample_df):
    result = analyze_products(sample_df)
    ranked = result["all_products_ranked_by_change"]
    assert ranked[0]["product"] == "Product A"  # biggest drop first
    assert ranked[0]["change_pct"] == pytest.approx(-50.0, abs=0.1)


def test_kpi_summary_totals(sample_df):
    kpi = get_kpi_summary(sample_df)
    assert kpi["num_regions"] == 2
    assert kpi["num_products"] == 2
    assert kpi["total_revenue"] == pytest.approx(sample_df["Revenue"].sum())
    assert kpi["profit_margin_pct"] > 0
