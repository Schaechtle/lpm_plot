import altair
import polars as pl

from lpm_plot import plot_marginal_1d


def test_plot_marginal_1d_smoke():
    observed_df = pl.read_csv("tests/resources/hand-written-observed.csv")
    synthetic_df = pl.read_csv("tests/resources/hand-written-observed.csv")
    columns = ["foo", "bar", "quagga"]
    assert isinstance(
        plot_marginal_1d(observed_df, synthetic_df, columns),
        altair.vegalite.v5.api.VConcatChart,
    )
