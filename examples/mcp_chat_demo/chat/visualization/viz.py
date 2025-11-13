"""Visualization utilities for token timelines."""

import matplotlib.pyplot as plt
import pandas as pd


from lxml import etree
from datetime import datetime
import pandas as pd
from io import BytesIO
import tiktoken


import pandas as pd

from lxml import etree
from datetime import datetime
import pandas as pd
from io import BytesIO


def compute_event_frequency(xml_str: str, bin_size: str = "day") -> pd.DataFrame:
    """
    Compute frequency distribution of events by timestamp bin and event type.

    Args:
        xml_str (str): XML content as string.
        bin_size (str): One of 'day', 'hour', 'minute'. Determines binning granularity.

    Returns:
        pd.DataFrame: A dataframe with counts of events per time bin and event type.
    """
    assert bin_size in {
        "day",
        "hour",
        "minute",
    }, "bin_size must be 'day', 'hour', or 'minute'"

    # Parse using BytesIO to handle encoding declaration
    root = etree.parse(BytesIO(xml_str.encode("utf-8"))).getroot()

    records = []
    for entry in root.findall(".//entry"):
        timestamp_str = entry.get("timestamp")
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M")
        for event in entry.findall("event"):
            event_type = event.get("type")
            records.append((timestamp, event_type))

    df = pd.DataFrame(records, columns=["timestamp", "event_type"])

    # Bin the timestamps
    if bin_size == "day":
        df["bin"] = df["timestamp"].dt.floor("D")
    elif bin_size == "hour":
        df["bin"] = df["timestamp"].dt.floor("H")
    elif bin_size == "minute":
        df["bin"] = df["timestamp"].dt.floor("T")

    # Count by bin and event_type
    counts = df.groupby(["bin", "event_type"]).size().unstack(fill_value=0)
    return counts


def aggregate_event_totals(counts_df: pd.DataFrame, level: str = "month") -> dict:
    """
    Aggregate a time-indexed event‐count table to day, month, or year totals.

    Args:
        counts_df:
            - If DataFrame: index must be a DatetimeIndex (each bin: day/hour/etc.),
              columns are event types and values are counts.
            - If Series: a single column of counts indexed by datetime.
        level: One of {"day", "month", "year"}. Controls aggregation bin.

    Returns:
        A dict mapping the period start (pd.Timestamp) → total event count.
    """
    # 1) ensure we have a DataFrame with a single "count" column
    if isinstance(counts_df, pd.Series):
        df = counts_df.to_frame(name="count")
    else:
        df = counts_df.copy()
        df["count"] = df.sum(axis=1)  # collapse across event types

    # 2) pick the right resample rule
    rule = {"day": "D", "month": "M", "year": "Y"}.get(level)
    if rule is None:
        raise ValueError("level must be one of 'day', 'month', or 'year'")

    # 3) resample and sum
    agg = df["count"].resample(rule).sum()

    return agg.to_dict()


# monthly_totals = aggregate_event_totals(df, level="month")


def compute_entry_token_lengths(
    xml_str: str, bin_size: str = "day", model: str = "gpt-3.5-turbo"
) -> pd.DataFrame:
    """
    Compute total token length of each XML <entry>, binned by day or year.

    Args:
        xml_str (str): XML content as string.
        bin_size (str): One of 'day' or 'year'. Determines binning granularity.
        model (str): The OpenAI model name to select tiktoken encoding.

    Returns:
        pd.DataFrame: A dataframe indexed by bin (Timestamp) with a column
                      'token_length' giving the sum of tokens for all entries in that bin.
    """
    assert bin_size in {"day", "year"}, "bin_size must be 'day' or 'year'"

    # load the right encoding for your model
    enc = tiktoken.encoding_for_model(model)

    # parse XML
    root = etree.parse(BytesIO(xml_str.encode("utf-8"))).getroot()

    records = []
    for entry in root.findall(".//entry"):
        # parse timestamp
        ts = entry.get("timestamp")
        timestamp = datetime.strptime(ts, "%Y-%m-%d %H:%M")

        # serialize this entry's XML and count tokens
        entry_xml = etree.tostring(entry, encoding="utf-8", method="xml").decode(
            "utf-8"
        )
        token_length = len(enc.encode(entry_xml))

        records.append((timestamp, token_length))

    df = pd.DataFrame(records, columns=["timestamp", "token_length"])

    # floor timestamps into bins
    if bin_size == "day":
        df["bin"] = df["timestamp"].dt.floor("D")
    else:  # bin_size == "year"
        # floor to Jan 1 of that year
        df["bin"] = df["timestamp"].dt.to_period("Y").dt.to_timestamp()

    # sum token lengths per bin
    result = df.groupby("bin")["token_length"].sum().to_frame()

    return result


def plot_token_timeline(
    df: pd.DataFrame,
    highlight_df: pd.DataFrame = None,
    bin_size: str = "day",
    bar_height: float = 1.0,
    gap: float = 0.1,
    bar_color: str = "#cccccc",
    highlight_color: str = "blue",  # 2A9D8F
    edge_color: str = "white",
    edge_width: float = 0.5,
    min_year_width_ratio: float = 0.01,
    fig_width: float = 10,
    fig_height: float = 2,
    year_line_offset_frac: float = 0.1,
    year_line_pad_frac: float = 0.02,
    tick_font_weight: str = "normal",  # new: e.g. "ultralight","light","normal","bold"
    dot_color: str = "#444",
    font_family: str = "sans-serif",  # new parameter for font family sans-serif
    query_datetime: pd.Timestamp = None,  # new parameter for query datetime
    dim_future: bool = True,  # new parameter to control dimming of future boxes
):
    """
    1-row timeline of token-length bricks:
      • all bins in `bar_color`, highlights overplotted in `highlight_color`
      • white gaps for non-contiguous bins
      • horizontal line segments below marking each full-year span
      • segments padded by `year_line_pad_frac * span` on each side
      • drawn at `bar_bottom - year_line_offset_frac * bar_height`
      • full-year ticks labeled with year; narrow years get a colored "•"
      • x-axis label weight set by `tick_font_weight`
      • figure size = (fig_width × fig_height)
      • font family can be specified with `font_family`
      • query datetime shown as vertical red line if provided
      • boxes after query datetime are dimmed if dim_future is True
    """
    # 1) sort & extract
    df = df.sort_index()
    times = df.index
    lengths = df["token_length"].values

    # 2) contiguity mask
    contig = [False]
    for prev, curr in zip(times[:-1], times[1:]):
        if bin_size == "day":
            contig.append((curr - prev) == pd.Timedelta(days=1))
        elif bin_size == "month":
            contig.append(curr.to_period("M") == prev.to_period("M") + 1)
        elif bin_size == "year":
            contig.append(curr.year == prev.year + 1)
        else:
            raise ValueError("bin_size must be 'day','month',or 'year'")

    # 3) compute x positions (with gaps)
    x_positions = []
    x = 0.0
    for is_cont, w in zip(contig, lengths):
        if not is_cont and x_positions:
            x += gap
        x_positions.append(x)
        x += w
    total_end = x_positions[-1] + lengths[-1]

    # 4) lookup maps
    pos_map = dict(zip(times, x_positions))
    width_map = dict(zip(times, lengths))

    # 5) compute each year's span
    years = sorted({t.year for t in times})
    year_spans = {}
    for yr in years:
        idxs = [i for i, t in enumerate(times) if t.year == yr]
        if not idxs:
            continue
        start = x_positions[idxs[0]]
        end = x_positions[idxs[-1]] + lengths[idxs[-1]]
        span = end - start
        year_spans[yr] = (start, end, span)

    # 6) y-coordinate for year lines (below the bars)
    line_y = -bar_height / 2 - (year_line_offset_frac * bar_height)

    # 7) plotting
    plt.rcParams["font.family"] = font_family
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_axisbelow(False)

    # draw all bins
    for t in times:
        # Determine opacity based on query datetime
        alpha = (
            0.25
            if (dim_future and query_datetime is not None and t > query_datetime)
            else 1.0
        )

        ax.barh(
            y=0,
            width=width_map[t],
            left=pos_map[t],
            height=bar_height,
            color=bar_color,
            edgecolor=edge_color,
            linewidth=edge_width,
            zorder=1,
            alpha=alpha,
        )

    # overplot highlights
    if highlight_df is not None:
        for t in highlight_df.sort_index().index:
            if t in pos_map:
                # Determine opacity based on query datetime
                alpha = (
                    0.25
                    if (
                        dim_future and query_datetime is not None and t > query_datetime
                    )
                    else 0.95
                )

                ax.barh(
                    y=0,
                    width=width_map[t],
                    left=pos_map[t],
                    height=bar_height,
                    color=highlight_color,
                    edgecolor=edge_color,
                    linewidth=edge_width,
                    zorder=2,
                    alpha=alpha,
                )

    # Add query datetime line if provided
    if query_datetime is not None:
        # Find the closest timestamp in our data
        closest_time = min(times, key=lambda x: abs(x - query_datetime))
        if closest_time in pos_map:
            x_pos = pos_map[closest_time]
            ax.axvline(
                x=x_pos, color="red", linestyle="-", linewidth=2, alpha=0.8, zorder=3
            )

    # 8) set up ticks & determine full years
    xticks, xtlabels = [], []
    full_years = []
    for yr in years:
        start, end, span = year_spans.get(yr, (None, None, 0))
        if span == 0:
            continue
        center = (start + end) / 2
        xticks.append(center)
        if span / total_end < min_year_width_ratio:
            xtlabels.append("•")
        else:
            xtlabels.append(str(yr))
            full_years.append(yr)

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtlabels)

    # 9) style each tick label: weight + dot color
    for lbl in ax.get_xticklabels():
        lbl.set_fontweight(tick_font_weight)
        if lbl.get_text() == "•":
            lbl.set_color(dot_color)

    # 10) draw padded line for each full year
    for yr in full_years:
        start, end, span = year_spans[yr]
        pad = span * year_line_pad_frac
        x0, x1 = start + pad, end - pad
        if x1 > x0:
            ax.hlines(
                y=line_y, xmin=x0, xmax=x1, color="black", linewidth=1.5, zorder=10
            )

    # 11) finalize
    ax.set_xlim(0, total_end)
    ax.margins(x=0)
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(axis="x", length=0)

    return fig
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":

    xml_str = open(
        "/users/jfries/code/lumia/data/collections/dev-corpus/135978074.xml", "r"
    ).read()
    tok_n = compute_entry_token_lengths(xml_str)
    highlight_df = tok_n.sample(n=150, random_state=22)
    fig = plot_token_timeline(
        tok_n,
        highlight_df=highlight_df,
        bin_size="day",
        fig_width=12,
        highlight_color="#1F78B4",
        min_year_width_ratio=0.05,
        fig_height=1.0,
        year_line_offset_frac=0.15,
        year_line_pad_frac=0.01,
        font_family="sans-serif",
    )
