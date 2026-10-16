
from pathlib import Path
import html

import pandas as pd
import plotly.express as px
import streamlit as st


# =====================================================
# CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Nexus AI | Analytics",
    page_icon="📊",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

SALES_FILE = PROJECT_DIR / "data" / "sales_metrics.csv"
REGION_FILE = PROJECT_DIR / "data" / "regional_sales.csv"


# =====================================================
# DESIGN
# =====================================================

st.markdown(
    """
    <style>
    @import url(
      'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap'
    );

    .stApp {
        background: #f7f8fc;
        font-family: 'DM Sans', sans-serif;
        color: #252832;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        letter-spacing: -0.7px;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .eyebrow {
        font-size: 10px;
        font-weight: 700;
        letter-spacing: 1.8px;
        color: #5b5ce2;
        text-transform: uppercase;
        margin-bottom: 9px;
    }

    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(28px, 3vw, 38px);
        font-weight: 800;
        letter-spacing: -1.4px;
        line-height: 1.2;
        color: #242735;
    }

    .hero-subtitle {
        color: #818797;
        font-size: 14px;
        line-height: 1.8;
        margin-top: 10px;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid #e8eaf1;
        border-radius: 15px;
        padding: 19px;
        min-height: 132px;
        box-shadow: 0 3px 12px rgba(20, 30, 60, 0.025);
    }

    .metric-label {
        color: #7c8393;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 12px;
    }

    .metric-value {
        color: #252832;
        font-family: 'Manrope', sans-serif;
        font-size: clamp(22px, 2vw, 29px);
        font-weight: 800;
        letter-spacing: -0.8px;
        overflow-wrap: anywhere;
    }

    .metric-note {
        color: #83899a;
        font-size: 11px;
        margin-top: 8px;
    }

    .panel {
        background: #ffffff;
        border: 1px solid #e8eaf1;
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 14px;
    }

    .panel-label {
        color: #818797;
        font-size: 10px;
        letter-spacing: 1.5px;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 8px;
    }

    .insight {
        padding: 15px;
        border: 1px solid #e8eaf1;
        border-radius: 12px;
        background: #ffffff;
        margin-bottom: 10px;
        line-height: 1.75;
        font-size: 13px;
        color: #555c6d;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
        border-bottom: 1px solid #e8eaf1;
    }

    .stTabs [data-baseweb="tab"] {
        height: 46px;
        font-weight: 600;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =====================================================
# LOAD AND VALIDATE DATA
# =====================================================

@st.cache_data
def load_data(sales_path, region_path, modified_sales, modified_regions):
    sales = pd.read_csv(sales_path)
    regions = pd.read_csv(region_path)

    required_sales = {"Quarter", "Revenue_USD"}
    required_regions = {"Region", "Revenue_USD"}

    if not required_sales.issubset(sales.columns):
        raise ValueError(
            "sales_metrics.csv needs Quarter and Revenue_USD columns."
        )

    if not required_regions.issubset(regions.columns):
        raise ValueError(
            "regional_sales.csv needs Region and Revenue_USD columns."
        )

    if sales.empty or regions.empty:
        raise ValueError("The CSV files must not be empty.")

    sales["Revenue_USD"] = pd.to_numeric(
        sales["Revenue_USD"],
        errors="raise",
    )

    regions["Revenue_USD"] = pd.to_numeric(
        regions["Revenue_USD"],
        errors="raise",
    )

    if sales["Revenue_USD"].isna().any():
        raise ValueError("Quarterly revenue contains missing values.")

    if regions["Revenue_USD"].isna().any():
        raise ValueError("Regional revenue contains missing values.")

    if (sales["Revenue_USD"] < 0).any():
        raise ValueError("Revenue cannot be negative in this dataset.")

    if (regions["Revenue_USD"] < 0).any():
        raise ValueError("Regional revenue cannot be negative.")

    if sales["Quarter"].duplicated().any():
        raise ValueError("Each quarter must appear only once.")

    if regions["Region"].duplicated().any():
        raise ValueError("Each region must appear only once.")

    if regions["Region"].isna().any():
        raise ValueError("Region names cannot be missing.")

    return sales, regions


if not SALES_FILE.exists() or not REGION_FILE.exists():
    st.error("Required CSV files were not found.")

    st.write("Expected locations:")

    st.code(
        "data/sales_metrics.csv\n"
        "data/regional_sales.csv"
    )

    st.stop()


try:
    sales, regions = load_data(
        str(SALES_FILE),
        str(REGION_FILE),
        SALES_FILE.stat().st_mtime,
        REGION_FILE.stat().st_mtime,
    )
except Exception as error:
    st.error(f"Unable to load the dashboard data: {error}")
    st.stop()


# =====================================================
# HEADER
# =====================================================

header_left, header_right = st.columns([5, 1.5])

with header_left:
    st.markdown(
        '<div class="eyebrow">NEXUS AI / BUSINESS INTELLIGENCE</div>'
        '<div class="hero-title">Analytics Overview</div>'
        '<div class="hero-subtitle">'
        'Monitor revenue performance, compare regions, and explore '
        'the numbers behind your business reports.'
        '</div>',
        unsafe_allow_html=True,
    )

with header_right:
    st.markdown(
        """
        <div style="text-align:right;padding-top:10px">
          <span style="
            background:#e7f7ee;
            color:#147c50;
            border-radius:20px;
            padding:8px 11px;
            font-size:10px;
            font-weight:700;">
            ● DATA LOADED
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")


# =====================================================
# FILTERS
# =====================================================

with st.expander("Dashboard filters", expanded=False):
    quarter_options = sales["Quarter"].tolist()

    selected_quarters = st.multiselect(
        "Select quarters",
        options=quarter_options,
        default=quarter_options,
    )

    region_options = regions["Region"].tolist()

    selected_regions = st.multiselect(
        "Select regions",
        options=region_options,
        default=region_options,
    )

filtered_sales = sales[
    sales["Quarter"].isin(selected_quarters)
].copy()

filtered_regions = regions[
    regions["Region"].isin(selected_regions)
].copy()


# =====================================================
# KEY PERFORMANCE INDICATORS
# =====================================================

quarter_values = sales["Quarter"].tolist()

if not quarter_values:
    st.warning("No quarterly data is available.")
    st.stop()

latest_quarter = quarter_values[-1]
latest_revenue = float(
    sales.loc[
        sales["Quarter"] == latest_quarter,
        "Revenue_USD",
    ].iloc[0]
)

growth_text = "Not available"
growth_color = "#858b98"
growth_note = "No previous quarter available"

if len(sales) >= 2:
    previous_quarter = quarter_values[-2]

    previous_revenue = float(
        sales.loc[
            sales["Quarter"] == previous_quarter,
            "Revenue_USD",
        ].iloc[0]
    )

    if previous_revenue != 0:
        growth = (
            (latest_revenue - previous_revenue)
            / previous_revenue
        ) * 100

        growth_text = f"{growth:+.1f}%"
        growth_color = "#15845a" if growth >= 0 else "#c54848"

        growth_note = (
            f"Compared with {previous_quarter}"
        )

top_region_name = "Not available"
top_region_revenue = 0.0

if not regions.empty:
    top_region = regions.loc[
        regions["Revenue_USD"].idxmax()
    ]

    top_region_name = str(top_region["Region"])
    top_region_revenue = float(top_region["Revenue_USD"])


def metric_card(label, value, note):
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{value}</div>
          <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


c1, c2, c3, c4 = st.columns(4)

with c1:
    metric_card(
        f"{latest_quarter} Revenue",
        f"${latest_revenue:,.0f}",
        "Latest quarter in dataset",
    )

with c2:
    metric_card(
        "Quarterly Change",
        growth_text,
        growth_note,
    )

with c3:
    metric_card(
        "Top Region",
        top_region_name,
        f"${top_region_revenue:,.0f} revenue",
    )

with c4:
    metric_card(
        "Regional Revenue",
        f"${regions['Revenue_USD'].sum():,.0f}",
        f"{len(regions)} regions in dataset",
    )


# =====================================================
# CHARTS
# =====================================================

st.write("")

tab_overview, tab_data, tab_insights = st.tabs(
    [
        "Overview",
        "Data Explorer",
        "Business Insights",
    ]
)


with tab_overview:

    chart_left, chart_right = st.columns(2)

    with chart_left:
        st.markdown(
            '<div class="panel-label">REVENUE PERFORMANCE</div>',
            unsafe_allow_html=True,
        )

        if filtered_sales.empty:
            st.info("Select at least one quarter.")
        else:
            fig_revenue = px.bar(
                filtered_sales,
                x="Quarter",
                y="Revenue_USD",
                text="Revenue_USD",
                labels={
                    "Quarter": "Quarter",
                    "Revenue_USD": "Revenue (USD)",
                },
                color_discrete_sequence=["#6264E8"],
            )

            fig_revenue.update_traces(
                texttemplate="$%{text:,.0f}",
                textposition="outside",
                cliponaxis=False,
                marker_cornerradius=6,
            )

            fig_revenue.update_layout(
                template="plotly_white",
                height=360,
                margin=dict(l=15, r=15, t=25, b=15),
                showlegend=False,
                yaxis=dict(
                    title="Revenue (USD)",
                    gridcolor="#edf0f5",
                    tickprefix="$",
                    separatethousands=True,
                ),
                xaxis=dict(title=""),
                font=dict(family="Arial", color="#51586a"),
            )

            st.plotly_chart(
                fig_revenue,
                use_container_width=True,
            )

    with chart_right:
        st.markdown(
            '<div class="panel-label">REGIONAL COMPARISON</div>',
            unsafe_allow_html=True,
        )

        if filtered_regions.empty:
            st.info("Select at least one region.")
        else:
            region_chart_data = filtered_regions.sort_values(
                "Revenue_USD",
                ascending=True,
            )

            fig_regions = px.bar(
                region_chart_data,
                x="Revenue_USD",
                y="Region",
                orientation="h",
                text="Revenue_USD",
                labels={
                    "Region": "Region",
                    "Revenue_USD": "Revenue (USD)",
                },
                color_discrete_sequence=["#8B83EF"],
            )

            fig_regions.update_traces(
                texttemplate="$%{text:,.0f}",
                textposition="outside",
                cliponaxis=False,
                marker_cornerradius=6,
            )

            fig_regions.update_layout(
                template="plotly_white",
                height=360,
                margin=dict(l=15, r=45, t=25, b=15),
                showlegend=False,
                xaxis=dict(
                    title="Revenue (USD)",
                    gridcolor="#edf0f5",
                    tickprefix="$",
                    separatethousands=True,
                ),
                yaxis=dict(title=""),
                font=dict(family="Arial", color="#51586a"),
            )

            st.plotly_chart(
                fig_regions,
                use_container_width=True,
            )

    st.caption(
        "Charts are calculated from the CSV datasets. "
        "They do not depend on generated AI answers."
    )


with tab_data:

    st.subheader("Quarterly revenue data")

    st.dataframe(
        filtered_sales,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download quarterly data",
        data=filtered_sales.to_csv(index=False).encode("utf-8"),
        file_name="filtered_sales_metrics.csv",
        mime="text/csv",
        use_container_width=False,
    )

    st.divider()

    st.subheader("Regional sales data")

    st.dataframe(
        filtered_regions.sort_values(
            "Revenue_USD",
            ascending=False,
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download regional data",
        data=filtered_regions.to_csv(index=False).encode("utf-8"),
        file_name="filtered_regional_sales.csv",
        mime="text/csv",
        key="download_regions",
    )


with tab_insights:

    st.subheader("Business findings")

    if len(sales) >= 2:
        previous_quarter = quarter_values[-2]

        previous_revenue = float(
            sales.loc[
                sales["Quarter"] == previous_quarter,
                "Revenue_USD",
            ].iloc[0]
        )

        difference = latest_revenue - previous_revenue

        if previous_revenue != 0:
            percent_change = (
                difference / previous_revenue
            ) * 100
        else:
            percent_change = None

        st.markdown(
            f"""
            <div class="insight">
              <b>Quarterly performance</b><br>
              Revenue changed from ${previous_revenue:,.0f}
              in {previous_quarter} to ${latest_revenue:,.0f}
              in {latest_quarter}.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if percent_change is not None:
            direction = (
                "increased"
                if difference > 0
                else "declined"
                if difference < 0
                else "remained unchanged"
            )

            st.markdown(
                f"""
                <div class="insight">
                  <b>Revenue movement</b><br>
                  Revenue {direction} by ${abs(difference):,.0f},
                  equivalent to {abs(percent_change):.2f}%.
                  This comparison does not establish the cause
                  of the change.
                </div>
                """,
                unsafe_allow_html=True,
            )

    if not regions.empty:
        highest = regions.loc[
            regions["Revenue_USD"].idxmax()
        ]

        lowest = regions.loc[
            regions["Revenue_USD"].idxmin()
        ]

        st.markdown(
            f"""
            <div class="insight">
              <b>Leading region</b><br>
            {html.escape(str(highest['Region']))}
              generated the highest recorded regional revenue:
              ${float(highest['Revenue_USD']):,.0f}.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="insight">
              <b>Lowest recorded region</b><br>
              {html.escape(str(lowest['Region']))}
              generated ${float(lowest['Revenue_USD']):,.0f}.
              Investigate this region's performance before
              deciding on operational changes.
            </div>
            """,
            unsafe_allow_html=True,
        )

        regional_total = float(regions["Revenue_USD"].sum())

        if abs(regional_total - latest_revenue) > 0.01:
            st.warning(
                "The sum of regional revenues does not match "
                "the latest quarterly revenue. Verify the datasets "
                "before interpreting the results."
            )
        else:
            st.success(
                "Data validation passed: regional revenue sums "
                "to the latest quarter's recorded revenue."
            )


# =====================================================
# FOOTER
# =====================================================

st.markdown(
    """
    <div style="
      text-align:center;
      padding-top:28px;
      color:#9298a6;
      font-size:11px;
      line-height:1.8;">
      NEXUS AI · ANALYTICS WORKSPACE<br>
      Findings describe the supplied sample data, not verified
      real-world business performance.
    </div>
    """,
    unsafe_allow_html=True,
)
