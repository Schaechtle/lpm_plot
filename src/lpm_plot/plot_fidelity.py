import polars as pl
import altair as alt

alt.data_transformers.enable("vegafusion")

METRICS = {
    "tvd": "Total variation distance",
    "kl": "Kullback–Leibler divergence",
    "js": "Jensen–Shannon divergence",
}
STROKEDASH = 5


def plot_fidelity(fidelity_df, metric="tvd"):
    # Convert Polars DataFrame to Pandas for compatibility with Altair
    fidelity_df_pandas = fidelity_df.to_pandas()

    # Altair visualization
    line_chart = (
        alt.Chart(fidelity_df_pandas)
        .mark_line(strokeDash=[STROKEDASH, STROKEDASH])
        .encode(
            x=alt.X("index:O"),  # ordinal scale, no axis labels
            y=alt.Y(f"{metric}:Q", title=METRICS.get(metric, metric)),
            color="model:N",
        )
    )

    point_chart = (
        alt.Chart(fidelity_df_pandas)
        .mark_point()
        .encode(
            x=alt.X(
                "index:O",
                title="Pairs of columns (ordered from best fit to worst)",
                axis=alt.Axis(labels=False),
            ),
            y=alt.Y(f"{metric}:Q"),
            color="model:N",
            tooltip=[
                alt.Tooltip("column-1:N", title="Column 1"),
                alt.Tooltip("column-2:N", title="Column 2"),
            ],
        )
    )

    # Combine the line chart and point chart into one layered chart
    final_chart = alt.layer(line_chart, point_chart).properties(width=400, height=400)
    return final_chart
