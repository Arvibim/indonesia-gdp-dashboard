import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from pathlib import Path
from libpysal.weights import Queen, KNN
from esda.moran import Moran, Moran_Local
from shapely import MultiPoint, box, voronoi_polygons

st.set_page_config(
    page_title="Indonesia GDP per Capita Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
    }
    .header-title {
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .header-subtitle {
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .section-divider {
        margin: 2rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="header-title">📊 Indonesia GDP per Capita Analysis Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="header-subtitle">Analyzing Regional Economic Growth across Indonesian Districts</div>',
    unsafe_allow_html=True
)
st.divider()

BASE_DIR = Path(__file__).resolve().parent
URL_CSV = BASE_DIR / "gdp_pc.csv"
URL_GEOJSON = BASE_DIR / "mapIndonesia514_new.geojson"

def file_signature(path):
    return path.stat().st_mtime_ns

@st.cache_data
def load_data(url, signature):
    gdp_df = pd.read_csv(url)

    gdp_df['districtID'] = gdp_df['districtID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    dist_col = 'district_en' if 'district_en' in gdp_df.columns else gdp_df.columns[1]

    year_cols = [c for c in gdp_df.columns if 'grdp_pc_' in c or any(chr.isdigit() for chr in c)]
    gdp_long = gdp_df.melt(
        id_vars=['districtID', dist_col],
        value_vars=year_cols,
        var_name='year_raw',
        value_name='gdp_pc'
    )
    gdp_long.rename(columns={dist_col: 'district_en'}, inplace=True)
    gdp_long['year'] = gdp_long['year_raw'].str.extract(r'(\d+)').astype(int)
    gdp_long['gdp_pc'] = pd.to_numeric(gdp_long['gdp_pc'].astype(str).str.replace(',', ''), errors='coerce')
    gdp_long = gdp_long.dropna(subset=['district_en', 'gdp_pc'])
    gdp_long = gdp_long[~gdp_long['district_en'].str.lower().isin(['nan', ''])]

    def map_island(idx):
        if idx.startswith('1') or idx.startswith('2'):
            return 'Sumatra'
        if idx.startswith('3'):
            return 'Java'
        if idx.startswith('5'):
            return 'Bali & Nusa Tenggara'
        if idx.startswith('6'):
            return 'Kalimantan'
        if idx.startswith('7'):
            return 'Sulawesi'
        if idx.startswith('8'):
            return 'Maluku'
        if idx.startswith('9'):
            return 'Papua'
        return 'Other'

    gdp_long['island'] = gdp_long['districtID'].apply(map_island)

    province_map = {
        '11': 'Aceh', '12': 'North Sumatra', '13': 'West Sumatra', '14': 'Riau', '15': 'Jambi',
        '16': 'South Sumatra', '17': 'Bengkulu', '18': 'Lampung', '19': 'Bangka Belitung', '21': 'Riau Islands',
        '31': 'Jakarta', '32': 'West Java', '33': 'Central Java', '34': 'East Java', '35': 'Yogyakarta',
        '36': 'Banten', '51': 'Bali', '52': 'West Nusa Tenggara', '53': 'East Nusa Tenggara',
        '61': 'West Kalimantan', '62': 'Central Kalimantan', '63': 'South Kalimantan', '64': 'East Kalimantan',
        '65': 'North Kalimantan', '71': 'North Sulawesi', '72': 'Central Sulawesi', '73': 'South Sulawesi',
        '74': 'Southeast Sulawesi', '75': 'Gorontalo', '76': 'West Sulawesi', '81': 'Maluku', '82': 'North Maluku',
        '91': 'West Papua', '92': 'South West Papua', '94': 'Papua', '95': 'South Papua', '96': 'Central Papua', '97': 'Highland Papua'
    }
    gdp_long['province'] = gdp_long['districtID'].str[:2].map(province_map).fillna('Unknown')
    return gdp_long

@st.cache_data
def load_map(url, signature):
    map_gdf = gpd.read_file(url)

    if map_gdf.crs is None or map_gdf.crs != "EPSG:4326":
        map_gdf = map_gdf.to_crs("EPSG:4326")

    map_gdf['districtID'] = map_gdf['districtID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    map_gdf['geometry'] = map_gdf.simplify(0.005, preserve_topology=True)

    map_gdf = map_gdf.set_index('districtID')
    return json.loads(map_gdf[['geometry']].to_json())

@st.cache_data
def load_gdf_for_bounds(url, signature):
    gdf = gpd.read_file(url)
    if gdf.crs is None or gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    gdf['districtID'] = gdf['districtID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    return gdf

@st.cache_data
def load_gdf_for_esda(url, signature):
    gdf = gpd.read_file(url)
    if gdf.crs is None or gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    gdf['districtID'] = gdf['districtID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    return gdf[['districtID', 'geometry']].copy()

def get_map_bounds(gdf, district_ids):
    if not district_ids or len(district_ids) == 0:
        return {"lat": -2.0, "lon": 118.0}, 5.2

    filtered_gdf = gdf[gdf['districtID'].isin(district_ids)]
    if filtered_gdf.empty:
        return {"lat": -2.0, "lon": 118.0}, 5.2

    bounds = filtered_gdf.total_bounds
    center_lon = (bounds[0] + bounds[2]) / 2
    center_lat = (bounds[1] + bounds[3]) / 2

    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    max_range = max(width, height)

    if max_range > 20:
        zoom = 3.5
    elif max_range > 10:
        zoom = 4.2
    elif max_range > 5:
        zoom = 5.5
    else:
        zoom = 6.5

    return {"lat": center_lat, "lon": center_lon}, zoom

def build_lisa_labels(local_moran):
    labels = []
    for quadrant, p_value in zip(local_moran.q, local_moran.p_sim):
        if p_value > 0.05:
            labels.append('Not Significant')
        elif quadrant == 1:
            labels.append('High-High')
        elif quadrant == 2:
            labels.append('Low-High')
        elif quadrant == 3:
            labels.append('Low-Low')
        elif quadrant == 4:
            labels.append('High-Low')
        else:
            labels.append('Not Significant')
    return labels

def build_tessellation_polygons(gdf_subset):
    projected = gdf_subset[['districtID', 'geometry']].copy().to_crs(3857)
    projected['centroid'] = projected.geometry.centroid

    minx, miny, maxx, maxy = projected.total_bounds
    max_range = max(maxx - minx, maxy - miny)
    padding = max(max_range * 0.1, 10000)
    extent = box(minx - padding, miny - padding, maxx + padding, maxy + padding)

    centroid_geoms = projected['centroid'].tolist()
    voronoi = voronoi_polygons(MultiPoint(centroid_geoms), extend_to=extent)
    tessellation = gpd.GeoDataFrame(
        {'geometry': list(voronoi.geoms)},
        crs=projected.crs
    )

    centroid_gdf = gpd.GeoDataFrame(
        projected[['districtID']].copy(),
        geometry=projected['centroid'],
        crs=projected.crs
    )
    tessellation['cell_id'] = range(len(tessellation))
    centroid_gdf['centroid_id'] = range(len(centroid_gdf))

    matched = gpd.sjoin_nearest(
        tessellation,
        centroid_gdf,
        how='left',
        distance_col='distance_to_centroid'
    )
    tessellation_by_district = matched.dissolve(by='districtID', as_index=False)[['districtID', 'geometry']]
    return tessellation_by_district.to_crs(gdf_subset.crs)

def calculate_esda(gdf_subset, weight_method):
    if len(gdf_subset) < 3:
        return None, None, None, None, None, None

    tessellation_gdf = build_tessellation_polygons(gdf_subset)
    tessellation_gdf = tessellation_gdf.merge(
        gdf_subset[['districtID', 'district_en', 'gdp_pc']],
        on='districtID',
        how='inner'
    )

    queen_weights = Queen.from_dataframe(tessellation_gdf, use_index=False)
    if queen_weights.n < 3:
        return None, None, None, None, None, None

    avg_neighbors = float(np.mean(list(queen_weights.cardinalities.values())))

    if weight_method == 'KNN':
        knn_k = max(1, min(int(round(avg_neighbors)), len(tessellation_gdf) - 1))
        weights = KNN.from_dataframe(tessellation_gdf, k=knn_k, use_index=False)
        weight_label = f"Distance (KNN), k={knn_k}"
    else:
        weights = queen_weights
        weight_label = "Queen contiguity"

    weights.transform = 'r'
    values = tessellation_gdf['gdp_pc'].to_numpy()
    moran = Moran(values, weights)
    local_moran = Moran_Local(values, weights)
    spatial_lag = weights.sparse @ values
    tessellation_gdf['spatial_lag'] = spatial_lag
    tessellation_gdf['avg_neighbors'] = avg_neighbors
    return moran, local_moran, spatial_lag, tessellation_gdf, avg_neighbors, weight_label

def ordered_labels_by_id(frame, label_col):
    ordered = (
        frame[[label_col, 'districtID']]
        .dropna()
        .drop_duplicates()
        .groupby(label_col, as_index=False)['districtID']
        .min()
        .sort_values('districtID')
    )
    return ordered[label_col].tolist()

df = load_data(URL_CSV, file_signature(URL_CSV))
geojson = load_map(URL_GEOJSON, file_signature(URL_GEOJSON))
gdf_for_bounds = load_gdf_for_bounds(URL_GEOJSON, file_signature(URL_GEOJSON))

with st.sidebar:
    st.header("Filter Data")
    st.markdown("---")

    selected_module = st.radio(
        "Select Module",
        ["Spatial Distribution", "Historical Trend", "ESDA Analysis"]
    )
    st.markdown("")

    if selected_module == "Historical Trend":
        selected_year = int(df['year'].max())
    else:
        selected_year = st.slider(
            "Select Year",
            min_value=int(df['year'].min()),
            max_value=int(df['year'].max()),
            value=int(df['year'].max())
        )
        st.markdown("")

    island_order = ['Sumatra', 'Java', 'Bali & Nusa Tenggara', 'Kalimantan', 'Sulawesi', 'Maluku', 'Papua']
    islands_unique = df['island'].unique()
    islands_sorted = [i for i in island_order if i in islands_unique]
    islands = ["National"] + islands_sorted
    selected_island = st.selectbox(
        "Select Island",
        islands,
        help="Choose 'National' to view all islands"
    )
    st.markdown("")

    if selected_island == "National":
        provinces_df = df[['province', 'districtID']].drop_duplicates().sort_values('districtID')
        provinces = provinces_df['province'].unique().tolist()
    else:
        provinces_df = df[df['island'] == selected_island][['province', 'districtID']].drop_duplicates().sort_values('districtID')
        provinces = provinces_df['province'].unique().tolist()

    selected_provinces = st.multiselect(
        "Select Province(s)",
        provinces,
        default=[],
        help="Leave empty to show all provinces"
    )
    st.markdown("")

    df_filtered_for_districts = df[df['year'] == selected_year]
    if selected_island != "National":
        df_filtered_for_districts = df_filtered_for_districts[df_filtered_for_districts['island'] == selected_island]
    if selected_provinces:
        df_filtered_for_districts = df_filtered_for_districts[df_filtered_for_districts['province'].isin(selected_provinces)]

    districts_df = df_filtered_for_districts[['district_en', 'districtID']].drop_duplicates().sort_values('districtID')
    districts = districts_df['district_en'].tolist()
    selected_districts = st.multiselect(
        "Select District(s)",
        districts,
        default=[],
        help="Leave empty to show all districts"
    )

df_year = df[df['year'] == selected_year]
if selected_island != "National":
    df_year = df_year[df_year['island'] == selected_island]
if selected_provinces:
    df_year = df_year[df_year['province'].isin(selected_provinces)]
if selected_districts:
    df_year = df_year[df_year['district_en'].isin(selected_districts)]

moran, local_moran, spatial_lag, weight_gdf = None, None, None, None
avg_neighbors = None
esda_weight_label = None
esda_gdf = None
if selected_module == "ESDA Analysis":
    gdf_for_esda = load_gdf_for_esda(URL_GEOJSON, file_signature(URL_GEOJSON))
    esda_gdf = gdf_for_esda.merge(df_year, on='districtID', how='inner')
    esda_weight_method = st.radio(
        "ESDA Weight Type",
        ["Queen contiguity", "Distance (KNN)"],
        index=0,
        horizontal=True
    )
    internal_weight_method = 'KNN' if esda_weight_method == "Distance (KNN)" else 'Queen'
    moran, local_moran, spatial_lag, weight_gdf, avg_neighbors, esda_weight_label = calculate_esda(esda_gdf, internal_weight_method)
    if local_moran is not None and weight_gdf is not None:
        weight_gdf['lisa_cluster'] = build_lisa_labels(local_moran)
        esda_gdf = esda_gdf.merge(weight_gdf[['districtID', 'lisa_cluster']], on='districtID', how='left')

if selected_module == "Spatial Distribution":
    st.markdown("## Key Statistics", unsafe_allow_html=True)
    st.caption("in Thousand Rupiah (Rp)")
    if not df_year.empty:
        metric_cols = st.columns(6, gap="medium")
        metrics_data = [
            ("Mean", f"Rp {df_year['gdp_pc'].mean():,.0f}", "📊"),
            ("Median", f"Rp {df_year['gdp_pc'].median():,.0f}", "📊"),
            ("Min", f"Rp {df_year['gdp_pc'].min():,.0f}", "📉"),
            ("Max", f"Rp {df_year['gdp_pc'].max():,.0f}", "📈"),
            ("Std Dev", f"Rp {df_year['gdp_pc'].std():,.0f}", "📊"),
            ("Districts", len(df_year), "🏘️"),
        ]
        for i, (label, value, icon) in enumerate(metrics_data):
            metric_cols[i].metric(f"{icon} {label}", value)
    else:
        st.warning("No data available for the selected filters.")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

if selected_module == "ESDA Analysis":
    st.markdown("## ESDA: Spatial Autocorrelation", unsafe_allow_html=True)
    if esda_weight_method == "Distance (KNN)":
        avg_text = f"{avg_neighbors:.2f}" if avg_neighbors is not None else "N/A"
        st.caption(f"Weights use Distance (KNN) with k based on the average Queen neighbors ({avg_text}).")
    else:
        st.caption("Weights use Queen contiguity on centroid-based tessellation polygons to avoid island observations.")
    if moran is not None and local_moran is not None and spatial_lag is not None:
        esda_cols = st.columns(5, gap="medium")
        esda_cols[0].metric("Global Moran's I", f"{moran.I:.4f}")
        esda_cols[1].metric("Permutation p-value", f"{moran.p_sim:.4f}")
        esda_cols[2].metric("Significant Districts", int((local_moran.p_sim <= 0.05).sum()))
        esda_cols[3].metric("Avg Neighbors", f"{avg_neighbors:.2f}" if avg_neighbors is not None else "N/A")
        esda_cols[4].metric("Weight Type", esda_weight_label if esda_weight_label is not None else "N/A")

        st.markdown("### LISA Cluster Map")
        esda_center, esda_zoom = get_map_bounds(gdf_for_bounds, esda_gdf['districtID'].tolist())
        fig_esda_map = px.choropleth_mapbox(
            esda_gdf,
            geojson=geojson,
            locations='districtID',
            color='lisa_cluster',
            hover_name='district_en',
            hover_data={'gdp_pc': ':,.0f', 'districtID': False, 'lisa_cluster': True},
            color_discrete_map={
                'High-High': '#d73027',
                'Low-Low': '#4575b4',
                'High-Low': '#f46d43',
                'Low-High': '#74add1',
                'Not Significant': '#bdbdbd'
            },
            mapbox_style="carto-positron",
            center=esda_center,
            zoom=esda_zoom,
            opacity=0.85,
            labels={'lisa_cluster': 'LISA Cluster', 'gdp_pc': 'GDP per Capita (Thousand Rp)'}
        )
        fig_esda_map.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=500,
            font=dict(size=11),
            legend_title_text='Cluster Type'
        )
        st.plotly_chart(fig_esda_map, use_container_width=True)

        st.markdown("### Moran Scatter Plot")
        fig_moran = px.scatter(
            weight_gdf,
            x='gdp_pc',
            y='spatial_lag',
            color='lisa_cluster',
            hover_name='district_en',
            custom_data=['district_en', 'gdp_pc', 'spatial_lag'],
            labels={'gdp_pc': 'GDP per Capita (Thousand Rp)', 'spatial_lag': 'Spatial Lag'},
            color_discrete_map={
                'High-High': '#d73027',
                'Low-Low': '#4575b4',
                'High-Low': '#f46d43',
                'Low-High': '#74add1',
                'Not Significant': '#bdbdbd'
            }
        )
        fig_moran.update_traces(
            hovertemplate="%{customdata[0]}<br>GDP per Capita=%{customdata[1]:,.0f} Thousand Rp<br>Spatial Lag=%{customdata[2]:,.0f}<extra></extra>"
        )
        fig_moran.add_hline(y=float(spatial_lag.mean()), line_dash='dash', line_color='gray')
        fig_moran.add_vline(x=float(weight_gdf['gdp_pc'].mean()), line_dash='dash', line_color='gray')
        fig_moran.update_layout(
            height=450,
            hovermode='closest',
            font=dict(size=11),
            legend_title_text='Cluster Type'
        )
        st.plotly_chart(fig_moran, use_container_width=True)
    else:
        st.info("ESDA needs at least 3 districts from the active filters to calculate Moran's I and LISA.")
elif selected_module == "Spatial Distribution":
    st.markdown("## Geographic & Trend Analysis", unsafe_allow_html=True)

    st.markdown("### Interactive Choropleth Map")
    if not df_year.empty:
        active_districts = df_year['districtID'].unique().tolist()
        map_center, map_zoom = get_map_bounds(gdf_for_bounds, active_districts)

        fig_map = px.choropleth_mapbox(
            df_year,
            geojson=geojson,
            locations='districtID',
            color='gdp_pc',
            hover_name='district_en',
            hover_data={'gdp_pc': ':,.0f', 'districtID': False},
            color_continuous_scale="Viridis",
            mapbox_style="carto-positron",
            center=map_center,
            zoom=map_zoom,
            opacity=0.85,
            labels={'gdp_pc': 'GDP per Capita (Thousand Rp)'}
        )
        fig_map.update_layout(
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            height=550,
            font=dict(size=11)
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("No data to display. Please select filters.")

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

    st.markdown("## Distribution & Rankings", unsafe_allow_html=True)

    if not df_year.empty:
        st.markdown("### GDP Distribution Histogram")
        hist_color = "island" if selected_island == "National" else "province"
        hist_legend_title = "Island" if selected_island == "National" else "Province"
        fig_hist = px.histogram(
            df_year,
            x="gdp_pc",
            color=hist_color,
            nbins=15,
            barmode="stack",
            color_discrete_sequence=[
                '#8FB9D4',
                '#7FA6C8',
                '#B7C9D6',
                '#A7D0C7',
                '#C8D6E5',
                '#9DB6C8',
                '#D6C9B8'
            ],
            labels={
                'gdp_pc': 'GDP per Capita (Thousand Rp)',
                'count': 'Number of Districts',
                'island': 'Island',
                'province': 'Province'
            },
            title=""
        )
        fig_hist.update_traces(hovertemplate="%{fullData.name}<br>count = %{y} districts<extra></extra>")
        fig_hist.update_layout(
            height=450,
            hovermode='closest',
            font=dict(size=11),
            legend_title_text=hist_legend_title
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        st.markdown("### Top 10 & Bottom 10 Districts")
        df_sorted = df_year.sort_values('gdp_pc', ascending=False)
        col4, col5 = st.columns(2, gap="large")

        with col4:
            st.markdown("### Top 10 Districts")
            top_10 = df_sorted[['district_en', 'province', 'island', 'gdp_pc']].head(10).copy()
            top_10['GDP per Capita\n(in Thousand Rp)'] = top_10['gdp_pc'].apply(lambda x: f"{x:,.0f}")
            display_df = top_10[['district_en', 'province', 'island', 'GDP per Capita\n(in Thousand Rp)']].rename(columns={'district_en': 'District', 'province': 'Province', 'island': 'Island'})
            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                height=400
            )

        with col5:
            st.markdown("### Bottom 10 Districts")
            bottom_10 = df_sorted[['district_en', 'province', 'island', 'gdp_pc']].tail(10).iloc[::-1].copy()
            bottom_10['GDP per Capita\n(in Thousand Rp)'] = bottom_10['gdp_pc'].apply(lambda x: f"{x:,.0f}")
            display_df = bottom_10[['district_en', 'province', 'island', 'GDP per Capita\n(in Thousand Rp)']].rename(columns={'district_en': 'District', 'province': 'Province', 'island': 'Island'})
            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True,
                height=400
            )
    else:
        st.warning("No data available for the selected filters.")
elif selected_module == "Historical Trend":
    st.markdown("## Historical Trend", unsafe_allow_html=True)

    st.markdown("### All Districts Trend (Log Scale)")
    st.caption("The trend is calculated on a log10 scale to make differences easier to compare, while the Y-axis labels are shown in the original GDP per capita values (Thousand Rp).")
    df_trend_all = df.copy()
    if selected_island != "National":
        df_trend_all = df_trend_all[df_trend_all['island'] == selected_island]
    if selected_provinces:
        df_trend_all = df_trend_all[df_trend_all['province'].isin(selected_provinces)]
    if selected_districts:
        df_trend_all = df_trend_all[df_trend_all['district_en'].isin(selected_districts)]

    df_trend_all = df_trend_all[df_trend_all['gdp_pc'] > 0].copy()
    if not df_trend_all.empty:
        if selected_island == "National" and not selected_provinces:
            highlight_level_options = ["No highlight", "Highlight Island", "Highlight Province", "Highlight District"]
        elif selected_provinces:
            highlight_level_options = ["No highlight", "Highlight District"]
        else:
            highlight_level_options = ["No highlight", "Highlight Province", "Highlight District"]

        selected_highlight_level = st.selectbox(
            "Highlight Trend By",
            highlight_level_options,
            index=0
        )

        if selected_highlight_level == "Highlight Island":
            highlight_item_options = ordered_labels_by_id(df_trend_all, 'island')
            selected_highlight_item = st.selectbox("Select Island to Highlight", highlight_item_options, index=0)
            highlight_column = 'island'
        elif selected_highlight_level == "Highlight Province":
            highlight_item_options = ordered_labels_by_id(df_trend_all, 'province')
            selected_highlight_item = st.selectbox("Select Province to Highlight", highlight_item_options, index=0)
            highlight_column = 'province'
        elif selected_highlight_level == "Highlight District":
            highlight_item_options = (
                df_trend_all[['district_en', 'districtID']]
                .dropna()
                .drop_duplicates()
                .sort_values('districtID')['district_en']
                .tolist()
            )
            selected_highlight_item = st.selectbox("Select District to Highlight", highlight_item_options, index=0)
            highlight_column = 'district_en'
        else:
            selected_highlight_item = None
            highlight_column = None

        if highlight_column is not None:
            highlighted_districts = df_trend_all.loc[
                df_trend_all[highlight_column] == selected_highlight_item,
                'district_en'
            ].dropna().unique().tolist()
        else:
            highlighted_districts = []

        df_trend_all['log_gdp_pc'] = np.log10(df_trend_all['gdp_pc'])
        df_trend_average = (
            df_trend_all.groupby('year', as_index=False)['log_gdp_pc']
            .mean()
            .sort_values('year')
        )
        df_trend_median = (
            df_trend_all.groupby('year', as_index=False)['log_gdp_pc']
            .median()
            .sort_values('year')
        )

        fig_trend_all = px.line(
            df_trend_all,
            x='year',
            y='log_gdp_pc',
            color='district_en',
            markers=False,
            title="",
            custom_data=['district_en', 'gdp_pc'],
            labels={'log_gdp_pc': 'Log10 GDP per Capita', 'year': 'Year', 'district_en': 'District'}
        )
        for trace in fig_trend_all.data:
            trace.hovertemplate = "%{customdata[0]}<br>Year=%{x}<br>GDP per Capita=%{customdata[1]:,.0f} Thousand Rp<extra></extra>"
            trace.showlegend = False
            if trace.name in highlighted_districts:
                trace.line.color = 'rgba(155, 93, 229, 1.0)'
                trace.line.width = 1.6
                trace.opacity = 1.0
            else:
                trace.line.color = 'rgba(99, 110, 114, 0.20)'
                trace.line.width = 1.0
                trace.opacity = 0.8
        fig_trend_all.add_trace(
            go.Scatter(
                x=df_trend_average['year'],
                y=df_trend_average['log_gdp_pc'],
                mode='lines',
                name='Average',
                line=dict(color='rgba(255, 174, 96, 0.98)', width=2.6, dash='dash'),
                hovertemplate='Average<br>Year=%{x}<br>Log10 GDP per Capita=%{y:.3f}<extra></extra>',
                showlegend=True
            )
        )
        fig_trend_all.add_trace(
            go.Scatter(
                x=df_trend_median['year'],
                y=df_trend_median['log_gdp_pc'],
                mode='lines',
                name='Median',
                line=dict(color='rgba(140, 215, 255, 1.0)', width=2.9, dash='dot'),
                hovertemplate='Median<br>Year=%{x}<br>Log10 GDP per Capita=%{y:.3f}<extra></extra>',
                showlegend=True
            )
        )
        fig_trend_all.update_layout(
            height=760,
            hovermode='closest',
            font=dict(size=10),
            showlegend=True,
            legend_title_text='Legend',
            margin=dict(l=20, r=20, t=30, b=30),
            yaxis_title='GDP per Capita (Thousand Rp)'
        )
        y_min = float(np.floor(df_trend_all['log_gdp_pc'].min() * 2) / 2)
        y_max = float(np.ceil(df_trend_all['log_gdp_pc'].max() * 2) / 2)
        y_tickvals = list(np.arange(y_min, y_max + 0.001, 0.5))
        y_ticktext = [f"{int(round(10 ** tick)):,}" for tick in y_tickvals]
        fig_trend_all.update_yaxes(tickmode='array', tickvals=y_tickvals, ticktext=y_ticktext)
        st.plotly_chart(fig_trend_all, use_container_width=True)
    else:
        st.info("No log-scale trend data available.")
