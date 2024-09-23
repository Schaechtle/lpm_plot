import polars as pl
import altair as alt

OBSERVED_COLOR = "#000000"
SYNTHETIC_COLOR = "#f28e2b"


def get_max_frequency(column, data):
    # Group by `data_source` and the given `column`, then count occurrences
    result = (
        data.group_by(["data_source", column])
        .agg(pl.count(column).alias("count"))  # Count occurrences of each value
        .group_by("data_source")
        .agg(pl.max("count").alias("max_count"))  # Get max count for each data_source
    )
    # Return the max of the max counts
    return result.select(pl.max("max_count")).item()


def plot_marginal_1d(observed_df, synthetic_df, columns):
    assert len(columns) > 0.0
    for c in columns:
        assert c in observed_df.columns, "column not in observed data"
        assert c in synthetic_df.columns, "column not in synthetic data"

    observed_df = observed_df[columns].with_columns(pl.lit("observed").alias("data_source"))
    synthetic_df = synthetic_df[columns].with_columns(pl.lit("synthetic").alias("data_source"))

    data = pl.concat([observed_df, synthetic_df])

    # XXX: horrible hack; but Altair doesn't allow me to add a custom legend.
    dummy_data_for_legend = pl.DataFrame(
        {
            "category": ["Observed", "Synthetic"],
            "dummy": [0, 0],  # Dummy column to avoid visible marks
        }
    ).to_pandas()

    # Create an empty plot with a legend and no visible marks
    legend = (
        alt.Chart(dummy_data_for_legend)
        .mark_point(size=0, opacity=0)
        .encode(
            color=alt.Color(
                "category:N",
                scale=alt.Scale(
                    domain=["Observed", "Synthetic"],
                    range=[OBSERVED_COLOR, SYNTHETIC_COLOR],
                ),
                legend=alt.Legend(title="Legend", symbolStrokeWidth=4),
            )
        )
    )

    def create_marginal_chart(field, color, max_count):
        return (
            alt.Chart(data.to_pandas())
            .mark_bar(color=color)
            .encode(
                x=alt.X(
                    "count():Q",
                    scale=alt.Scale(domain=[0, max_count]),
                    axis=alt.Axis(orient="top"),
                ),
                y=alt.Y(
                    f"{field}:N",
                    axis=alt.Axis(
                        titleAnchor="start",
                        titleAlign="right",
                        titlePadding=1,
                        titleAngle=0,  # Rotate the y-axis label by 90 degrees (change as needed)
                    ),
                ),
                color=alt.value(color),
            )
        )

    # Creating the charts for observed and synthetic collections
    def create_comparison(column, data):
        max_count = get_max_frequency(column, data)
        chart_observed = create_marginal_chart(
            column, OBSERVED_COLOR, max_count
        ).transform_filter(alt.datum.data_source == "observed")
        chart_synthetic = create_marginal_chart(
            column, SYNTHETIC_COLOR, max_count
        ).transform_filter(alt.datum.data_source == "synthetic")
        return alt.hconcat(chart_observed, chart_synthetic)

    one_d_plots = [create_comparison(column, data) for column in columns]
    # Layer the charts for observed and synthetic data points
    combined_chart = (
        alt.vconcat(*one_d_plots, legend)
        .resolve_scale(color="independent")
        .properties(title="1-D Marginals")
    )
    return combined_chart
