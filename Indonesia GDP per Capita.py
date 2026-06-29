import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import json

# ⚙️ KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Indonesia GDP per Capita Dashboard",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# 🎨 CUSTOM STYLING
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

# Header dengan styling
st.markdown(
    '<div class="header-title">📊 Indonesia GDP per Capita Analysis Dashboard</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="header-subtitle">Analyzing Regional Economic Growth across Indonesian Districts</div>',
    unsafe_allow_html=True
)
st.divider()

# 🔗 RAW DATA URL DARI GITHUB
URL_CSV = "https://raw.githubusercontent.com/Arvibim/indonesia514/d6b79ac30df93021fcdb9ed1f2ca970655263a67/gdp/gdp_pc.csv"
URL_GEOJSON = "https://raw.githubusercontent.com/Arvibim/indonesia514/d6b79ac30df93021fcdb9ed1f2ca970655263a67/maps/mapIndonesia514_new.geojson"

# 📥 PEMROSESAN DATA (DI-CACHE AGAR CEPAT)
@st.cache_data
def load_data(url):
    gdp_df = pd.read_csv(url)
    
    # 🔧 FIX 1: Hapus akhiran desimal '.0' dan spasi kosong di ID
    gdp_df['districtID'] = gdp_df['districtID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    dist_col = 'district_en' if 'district_en' in gdp_df.columns else gdp_df.columns[1]
    
    # Melt Data
    year_cols = [c for c in gdp_df.columns if 'grdp_pc_' in c or any(chr.isdigit() for chr in c)]
    gdp_long = gdp_df.melt(id_vars=['districtID', dist_col], value_vars=year_cols, var_name='year_raw', value_name='gdp_pc')
    gdp_long.rename(columns={dist_col: 'district_en'}, inplace=True)
    gdp_long['year'] = gdp_long['year_raw'].str.extract(r'(\d+)').astype(int)
    gdp_long['gdp_pc'] = pd.to_numeric(gdp_long['gdp_pc'].astype(str).str.replace(',', ''), errors='coerce')
    gdp_long = gdp_long.dropna(subset=['district_en', 'gdp_pc'])
    gdp_long = gdp_long[~gdp_long['district_en'].str.lower().isin(['nan', ''])]

    # Pola pengelompokan pulau (termasuk Kep. Riau berawalan 2)
    def map_island(idx):
        if idx.startswith('1') or idx.startswith('2'): return 'Sumatra'
        if idx.startswith('3'): return 'Java'
        if idx.startswith('5'): return 'Bali & Nusa Tenggara'
        if idx.startswith('6'): return 'Kalimantan'
        if idx.startswith('7'): return 'Sulawesi'
        if idx.startswith('8'): return 'Maluku'
        if idx.startswith('9'): return 'Papua'
        return 'Other'
    
    gdp_long['island'] = gdp_long['districtID'].apply(map_island)
    
    # Tambahkan kolom provinsi berdasarkan 2 digit pertama districtID
    province_map = {
        '11': 'Aceh', '12': 'North Sumatra', '13': 'West Sumatra', '14': 'Riau', '15': 'Jambi', 
        '16': 'South Sumatra', '17': 'Bengkulu', '18': 'Lampung', '19': 'Bangka Belitung', '21': 'Riau Islands',
        '31': 'Jakarta', '32': 'West Java', '33': 'Central Java', '34': 'East Java', '35': 'Yogyakarta',
        '36': 'Banten', '51': 'Bali', '52': 'West Nusa Tenggara', '53': 'East Nusa Tenggara',
        '61': 'West Kalimantan', '62': 'Central Kalimantan', '63': 'South Kalimantan', '64': 'East Kalimantan',
        '65': 'North Kalimantan', '71': 'North Sulawesi', '72': 'Central Sulawesi', '73': 'South Sulawesi',
        '74': 'Southeast Sulawesi', '75': 'Gorontalo', '76': 'West Sulawesi', '81': 'Maluku', '82': 'North Maluku',
        '91': 'Papua', '92': 'West Papua', '94': 'Papua Central', '95': 'Papua Pegunungan', '96': 'Papua Jaya'
    }
    gdp_long['province'] = gdp_long['districtID'].str[:2].map(province_map).fillna('Unknown')
    return gdp_long

@st.cache_data
def load_map(url):
    map_gdf = gpd.read_file(url)
    
    # 🔧 FIX 2: Format CRS dan index GeoJSON untuk Mapbox Plotly
    if map_gdf.crs is None or map_gdf.crs != "EPSG:4326":
        map_gdf = map_gdf.to_crs("EPSG:4326")
        
    map_gdf['districtID'] = map_gdf['districtID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    map_gdf['geometry'] = map_gdf.simplify(0.005, preserve_topology=True)
    
    map_gdf = map_gdf.set_index('districtID')
    return json.loads(map_gdf[['geometry']].to_json())

@st.cache_data
def load_gdf_for_bounds(url):
    """Load GeoDataFrame untuk bounds calculation"""
    gdf = gpd.read_file(url)
    if gdf.crs is None or gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    gdf['districtID'] = gdf['districtID'].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    return gdf

def get_map_bounds(gdf, district_ids):
    """Calculate map center and zoom from district bounds"""
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

# Panggil fungsi
df = load_data(URL_CSV)
geojson = load_map(URL_GEOJSON)
gdf_for_bounds = load_gdf_for_bounds(URL_GEOJSON)

# 🎛️ SIDEBAR: KONTROL FILTER
with st.sidebar:
    st.header("🔍 Filter Data")
    st.markdown("---")
    
    selected_year = st.slider(
        "Select Year",
        min_value=int(df['year'].min()),
        max_value=int(df['year'].max()),
        value=int(df['year'].max())
    )
    st.markdown("")
    
    # Filter Pulau (sorted by islandID)
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
    
    # Filter Provinsi (Multi-select) - sorted by districtID
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
    
    # Filter Kabupaten/Kota (Multi-select) - sorted by districtID
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

# 🔄 FILTERING DATA
df_year = df[df['year'] == selected_year]
if selected_island != "National":
    df_year = df_year[df_year['island'] == selected_island]
if selected_provinces:
    df_year = df_year[df_year['province'].isin(selected_provinces)]
if selected_districts:
    df_year = df_year[df_year['district_en'].isin(selected_districts)]

# 📈 METRIK DESKRIPTIF
st.markdown("## 📈 Key Statistics", unsafe_allow_html=True)
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

# 🗺️ VISUALISASI UTAMA
st.markdown("## 🗺️ Geographic & Trend Analysis", unsafe_allow_html=True)

# Interactive Choropleth Map (Full Width)
st.markdown("### Interactive Choropleth Map")
if not df_year.empty:
    # Calculate dynamic map center and zoom based on active filters
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
        margin={"r":0,"t":0,"l":0,"b":0},
        height=550,
        font=dict(size=11)
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No data to display. Please select filters.")

# Historical Trend (Full Width)
st.markdown("### Historical Trend (Top 10 & Bottom 10)")
if not df_year.empty and len(df_year) > 0:
    df_sorted = df_year.sort_values('gdp_pc', ascending=False)
    top_bottom_districts = pd.concat([df_sorted.head(10), df_sorted.tail(10)])['districtID']
    
    df_trend = df[(df['districtID'].isin(top_bottom_districts))]
    if not df_trend.empty:
        fig_trend = px.line(
            df_trend,
            x='year',
            y='gdp_pc',
            color='district_en',
            markers=True,
            title="",
            labels={'gdp_pc': 'GDP per Capita (Thousand Rp)', 'year': 'Year', 'district_en': 'District'}
        )
        fig_trend.update_layout(
            height=500,
            hovermode='x unified',
            font=dict(size=10),
            legend=dict(font=dict(size=9))
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.info("No trend data available.")
else:
    st.info("No data to display. Please select filters.")

st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

# 📊 HISTOGRAM & TABEL PERINGKAT
st.markdown("## 📊 Distribution & Rankings", unsafe_allow_html=True)

if not df_year.empty:
    col3, col4, col5 = st.columns([2, 1, 1], gap="large")
    
    with col3:
        st.markdown("### GDP Distribution Histogram")
        fig_hist = px.histogram(
            df_year,
            x="gdp_pc",
            nbins=15,
            color_discrete_sequence=['#1f77b4'],
            labels={'gdp_pc': 'GDP per Capita (Thousand Rp)', 'count': 'Number of Districts'},
            title=""
        )
        fig_hist.update_layout(
            height=400,
            showlegend=False,
            hovermode='x unified',
            font=dict(size=11)
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col4:
        st.markdown("### 🏆 Top 10 Districts")
        top_10 = df_sorted[['district_en', 'gdp_pc']].head(10).copy()
        top_10['GDP per Capita\n(in Thousand Rp)'] = top_10['gdp_pc'].apply(lambda x: f"{x:,.0f}")
        
        # Create styled dataframe
        display_df = top_10[['district_en', 'GDP per Capita\n(in Thousand Rp)']].rename(columns={'district_en': 'District'})
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            height=400
        )
    
    with col5:
        st.markdown("### ⚠️ Bottom 10 Districts")
        bottom_10 = df_sorted[['district_en', 'gdp_pc']].tail(10).iloc[::-1].copy()
        bottom_10['GDP per Capita\n(in Thousand Rp)'] = bottom_10['gdp_pc'].apply(lambda x: f"{x:,.0f}")
        
        # Create styled dataframe
        display_df = bottom_10[['district_en', 'GDP per Capita\n(in Thousand Rp)']].rename(columns={'district_en': 'District'})
        st.dataframe(
            display_df,
            hide_index=True,
            use_container_width=True,
            height=400
        )
else:
    st.warning("No data available for the selected filters.")