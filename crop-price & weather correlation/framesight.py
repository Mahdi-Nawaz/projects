import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="FARMSIGHT Crop & Weather Dashboard", layout="wide")
st.sidebar.image("logo.png", width=140)

# Sidebar styles
st.sidebar.markdown("""
    <style>
    .sidebar .sidebar-content {background-color: #f3f7fa; padding-top: 5px;}
    .sidebar img {display:block; margin-left:auto; margin-right:auto;}
    .sidebar .stSelectbox > label, .sidebar .stSlider > label {color: #004e92;}
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("🌾 FARMSIGHT navigation")
st.sidebar.markdown("""
- **Main Analysis**
- **Seasonal Trends**
- **Correlation Heatmap**
- **Insights**
- **Download**
- **About**
""")

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv("crop and weather new.csv")
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df['District Name'] = df['District Name'].str.strip().str.lower()
    df['Commodity'] = df['Commodity'].str.strip().str.lower()
    return df

df = load_data()

# --- TITLE ---
col1, col2 = st.columns([0.10, 0.88])
with col1:
    st.image("logo.png")
with col2:
    st.markdown("<h1 style='color:#004e92;text-align:center'> FARMSIGHT Crop & Weather Dashboard</h1>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; font-size:18px;'>
<b>Explore price-weather relationships for major crops across districts, seasons, and years.</b>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR FILTERS ---
districts = sorted(df['District Name'].dropna().unique())
commodities = sorted(df['Commodity'].dropna().unique())

selected_district = st.sidebar.selectbox('Choose District', districts)
selected_crop = st.sidebar.selectbox('Choose Crop', commodities)
date_range = st.sidebar.date_input(
    "Date Range",
    [df['Date'].min(), df['Date'].max()],
    min_value=df['Date'].min(),
    max_value=df['Date'].max()
)

df_filtered = df[
    (df['District Name'] == selected_district) &
    (df['Commodity'] == selected_crop) &
    (df['Date'] >= pd.to_datetime(date_range[0])) &
    (df['Date'] <= pd.to_datetime(date_range[1]))
].sort_values('Date')

# --- SUMMARY METRICS ---
with st.container():
    st.markdown("### 🔎 Key Figures")
    colm = st.columns(3)
    colm[0].metric("Avg. Price (Rs./Quintal)", f"{df_filtered['Modal Price (Rs./Quintal)'].mean():.1f}")
    colm[1].metric("Avg. Max Temp (°C)", f"{df_filtered['temperature_2m_max'].mean():.1f}")
    colm[2].metric("Total Rainfall (mm)", f"{df_filtered['precipitation_sum'].sum():.1f}")

# --- INTERACTIVE SEASONAL TRENDS ---
with st.container():
    st.markdown("### 📈 Seasonal Trends (Price, Temperature, Rainfall)")
    fig_trend = px.line(
        df_filtered,
        x="Date",
        y=["Modal Price (Rs./Quintal)", "temperature_2m_max", "precipitation_sum"],
        template="plotly_white",
        labels={"value": "Value", "variable": "Metric"},
        color_discrete_map={
            "Modal Price (Rs./Quintal)": "#0099e5",
            "temperature_2m_max": "#fdbb2d",
            "precipitation_sum": "#4CAF50"
        },
        title=f"{selected_crop.title()} Price, Temperature & Rainfall - {selected_district.title()}"
    )
    fig_trend.update_traces(mode="lines+markers")
    fig_trend.update_layout(hovermode="x unified", legend_title_text="Metric")
    st.plotly_chart(fig_trend, use_container_width=True)

# --- CORRELATION HEATMAP ---
with st.container():
    st.markdown("### 🔥 Correlation between Price and Weather")
    num_cols = ['Modal Price (Rs./Quintal)', 'temperature_2m_max', 'temperature_2m_min', 'precipitation_sum']
    corr = df_filtered[num_cols].corr().round(2)

    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        colorscale="YlGnBu",
        zmin=-1, zmax=1,
        text=corr.values,
        texttemplate="%{text}",
        showscale=True
    ))
    fig_corr.update_layout(title="Correlation Heatmap", template="plotly_white")
    st.plotly_chart(fig_corr, use_container_width=True)

# --- SCATTER PLOT ---
with st.container():
    st.markdown("### 🌦️ Price vs. Weather Feature")
    xvar = st.selectbox('Scatterplot Weather Variable', ['temperature_2m_max', 'temperature_2m_min', 'precipitation_sum'])
    fig_scatter = px.scatter(
        df_filtered,
        x=xvar,
        y="Modal Price (Rs./Quintal)",
        trendline="ols",
        color="Market Name",
        template="plotly_white",
        opacity=0.75,
        title=f"Price vs. {xvar.replace('_', ' ').title()}",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- MOVING AVERAGE ---
with st.container():
    st.markdown('### 🔄 Moving Average Trends')
    window = st.slider("Moving Average Window", 7, 60, 21)
    df_filtered['Price_MA'] = df_filtered['Modal Price (Rs./Quintal)'].rolling(window=window).mean()
    df_filtered['TempMax_MA'] = df_filtered['temperature_2m_max'].rolling(window=window).mean()
    df_filtered['Rain_MA'] = df_filtered['precipitation_sum'].rolling(window=window).mean()

    fig_ma = px.line(
        df_filtered,
        x="Date",
        y=["Price_MA", "TempMax_MA", "Rain_MA"],
        labels={"value": "Value", "variable": "Metric"},
        template="plotly_white",
        title=f"Moving Averages ({window} Days)"
    )
    fig_ma.update_layout(legend_title_text="Metric")
    st.plotly_chart(fig_ma, use_container_width=True)

# --- PRICE DISTRIBUTION ---
with st.container():
    st.markdown("### 📦 Price Distribution in Selection")
    fig_box = px.box(
        df_filtered,
        y="Modal Price (Rs./Quintal)",
        points="all",
        title=f"{selected_crop.title()} Price Distribution",
        template="plotly_white"
    )
    st.plotly_chart(fig_box, use_container_width=True)

# --- AVERAGE PRICE BY MARKET ---
with st.container():
    st.markdown("### 🏬 Average Price by Market in Selection")
    avg_market_price = (
        df_filtered.groupby("Market Name")["Modal Price (Rs./Quintal)"]
        .mean()
        .reset_index()
        .rename(columns={"Modal Price (Rs./Quintal)": "Average Price"})
    )
    fig_bar = px.bar(
        avg_market_price,
        x="Market Name",
        y="Average Price",
        title=f"Average Price for {selected_crop.title()} in {selected_district.title()}",
        template="plotly_white",
        color="Average Price",
        color_continuous_scale="YlGnBu"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# --- DATA TABLE ---
with st.container():
    st.markdown("### 📋 Raw Data Table (Hidden Date Column)")
    show_cols = [c for c in df_filtered.columns if c.lower() != 'date']
    st.dataframe(df_filtered[show_cols], use_container_width=True)

    if st.checkbox("Download Table (CSV)"):
        st.download_button(
            label="Download Current Table",
            data=df_filtered[show_cols].to_csv(index=False).encode(),
            file_name=f"{selected_crop}_{selected_district}_filtered.csv",
            mime='text/csv'
        )

# --- FOOTER ---
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:90%'>Made with💚for agri-insight • FARMSIGHT 025</div>",
    unsafe_allow_html=True
)
 
