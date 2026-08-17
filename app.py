import ee
import streamlit as st
import pandas as pd
import numpy as np
import folium
import plotly.express as px
from streamlit_folium import folium_static
from datetime import datetime, timedelta

# 1. Page Configuration
st.set_page_config(layout="wide", page_title="India Climate Resilient Suitability Engine")
st.title("🛰️ Rainfall Data and Slope Analysis Engine")
st.subheader("Decision Support System for Climate-Resilient Urban Planning")

# 2. Authenticate and Initialize GEE
project_id = 'climate-resilition'
try:
    ee.Initialize(project=project_id)
except Exception as e:
    st.error(f"Earth Engine Initialization Failed: {e}")
    st.stop()

# 3. Sidebar Interactive Controls
st.sidebar.header("🎛️ Analysis Matrices Configuration")
rain_weight = st.sidebar.slider("Satellite Rain Data Weight", 0.0, 1.0, 0.50, 0.05)
slope_weight = round(1.0 - rain_weight, 2)
st.sidebar.text(f"Topographic Slope Weight: {slope_weight}")

# ===========================================================================================================
# 🧭 GEOGRAPHIC BASELINE SETTINGS: DEHRADUN & MUSSOORIE (HIGH VARIETY)
# ===========================================================================================================
lat_min, lat_max = 30.15, 30.50
lon_min, lon_max = 77.80, 78.25
roi_bounds = ee.Geometry.Rectangle([lon_min, lat_min, lon_max, lat_max])
map_center = [30.32, 78.03] # Centered near Dehradun city

# Dynamic Rolling Time-Window
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
end_date_str = end_date.strftime('%Y-%m-%d')
start_date_str = start_date.strftime('%Y-%m-%d')


# ===========================================================================================================
# ⚡️ ENGINE 1: VISUAL LAYER GENERATION (STABLE 2D TILE)
# ==========================================================================================================
@st.cache_resource(show_spinner="Rendering Cloud Map Matrix...")
def generate_visual_tile(r_weight, s_weight):
    srtm_dem = ee.Image('USGS/SRTMGL1_003').clip(roi_bounds)
    slope = ee.Terrain.slope(srtm_dem)
    
    gpm_precipitation = (ee.ImageCollection('NASA/GPM_L3/IMERG_V07')
                         .filterBounds(roi_bounds)
                         .filterDate(start_date_str, end_date_str)
                         .select('precipitation')
                         .mean()
                         .clip(roi_bounds))

    esa_lulc = ee.ImageCollection("ESA/WorldCover/v200").mosaic().select('Map').clip(roi_bounds)
    non_built_mask = esa_lulc.neq(50) 

    # Dynamic tuning for mountainous regional risk scaling
    rain_safety = ee.Image(1.0).subtract(gpm_precipitation.divide(0.4)).clamp(0.0, 1.0)
    slope_safety = ee.Image(1.0).subtract(slope.divide(25.0)).clamp(0.0, 1.0)

    suitability = (rain_safety.multiply(r_weight)).add(slope_safety.multiply(s_weight))
    masked_suitability = suitability.updateMask(non_built_mask)
    
    viz_params = {
        'min': 0.35,
        'max': 0.85,
        'palette': ['#d73027', '#fdae61', '#1a9850'] # Red -> Yellow -> Green
    }
    map_id = masked_suitability.getMapId(viz_params)
    return map_id['tile_fetcher'].url_format


# =========================================================================================================
# ⚡️ ENGINE 2: HIGH-SPEED VECTOR SAMPLER FOR CHARTS & TABLES
# =========================================================================================================
@st.cache_data(show_spinner="Extracting Analytical Table Reports...")
def extract_numerical_matrix(r_weight, s_weight):
    srtm_dem = ee.Image('USGS/SRTMGL1_003').clip(roi_bounds)
    slope = ee.Terrain.slope(srtm_dem).rename('slope')
    
    gpm_precipitation = (ee.ImageCollection('NASA/GPM_L3/IMERG_V07')
                         .filterBounds(roi_bounds)
                         .filterDate(start_date_str, end_date_str)
                         .select('precipitation')
                         .mean()
                         .clip(roi_bounds)).rename('rain')

    esa_lulc = ee.ImageCollection("ESA/WorldCover/v200").mosaic().select('Map').clip(roi_bounds)
    non_built_mask = esa_lulc.neq(50) 

    rain_safety = ee.Image(1.0).subtract(gpm_precipitation.divide(0.4)).clamp(0.0, 1.0)
    slope_safety = ee.Image(1.0).subtract(slope.divide(25.0)).clamp(0.0, 1.0)

    suitability = (rain_safety.multiply(r_weight)).add(slope_safety.multiply(s_weight)).rename('score')
    
    zone_classes = (suitability.lt(0.48).multiply(1)
                    .add(suitability.gte(0.48).And(suitability.lt(0.68)).multiply(2))
                    .add(suitability.gte(0.68).multiply(3))
                    .rename('zone_class'))

    combined_stack = suitability.addBands(slope).addBands(gpm_precipitation).addBands(zone_classes).updateMask(non_built_mask)
    
    # Fast stratified sampling to align charts perfectly
    sample_points = combined_stack.stratifiedSample(
        numPoints=45,
        classBand='zone_class',
        region=roi_bounds,
        scale=3000,
        geometries=True
    )
    
    try:
        raw_features = sample_points.getInfo().get('features', [])
        rows = []
        for feat in raw_features:
            props = feat.get('properties', {})
            geom = feat.get('geometry', {})
            coords = geom.get('coordinates', [0.0, 0.0])
            
            if 'score' in props:
                rows.append({
                    'Latitude': round(coords[1], 4),  # Corrected indexing
                    'Longitude': round(coords[0], 4), # Corrected indexing
                    'Slope (Deg)': round(props.get('slope', 0.0), 1),
                    'Rain (mm/hr)': round(props.get('rain', 0.0), 3),
                    'Suitability Index': round(props.get('score', 0.0), 3)
                })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()

# Run both engines independently
tile_url = generate_visual_tile(rain_weight, slope_weight)
df_metrics = extract_numerical_matrix(rain_weight, slope_weight)

# =================================================================================================================================
# UI LAYOUT SYSTEM: DUAL COLUMN SPECIFICATIONS
# =================================================================================================================================
col1, col2 = st.columns([3, 2.5])

with col1:
    st.markdown("### 🌍 Geographic Risk & Suitability Grid")
    st.info(f"🛰️ **Map Status**: Stable 2D Overlay Active. Full-screen button available in the top-left corner of the map canvas.")
    
    # Initialize the base map canvas
    m = folium.Map(location=map_center, zoom_start=11, tiles='OpenStreetMap')
    
    # FIXED: Check and add the plugin safely without printing technical objects to the screen
    if hasattr(folium.plugins, 'Fullscreen'):
        folium.plugins.Fullscreen(
            position='topleft', 
            title='Fullscreen View', 
            title_cancel='Exit'
        ).add_to(m)
    
    # Overlay the Google Earth Engine satellite layers
    folium.TileLayer(
        tiles=tile_url,
        attr='Google Earth Engine / NASA / ESA',
        name='Suitability Matrix',
        overlay=True,
        opacity=0.75
    ).add_to(m)
    
    # Render the map frame inside your browser window
    folium_static(m, width=780, height=480)

with col2:
    st.markdown("### 📊 Zonal Distribution Profile")
    
    if not df_metrics.empty:
        total_cells = len(df_metrics)
        high_hazard = int(np.sum(df_metrics['Suitability Index'] < 0.48) / total_cells * 100)
        moderate_risk = int(np.sum((df_metrics['Suitability Index'] >= 0.48) & (df_metrics['Suitability Index'] < 0.68)) / total_cells * 100)
        optimal_safe = int(100 - high_hazard - moderate_risk)
        
        m_col1, m_col2 = st.columns(2)
        with m_col1:
            st.metric(label="⚠️ High Hazard Area", value=f"{high_hazard}%", delta="Unsuitable Ground", delta_color="inverse")
        with m_col2:
            st.metric(label="🟢 Highly Resilient Land", value=f"{optimal_safe}%", delta="Optimal Zone")
            
        st.write("---")
        
        # Perfect Color-Mapped Horizontal Bar Chart
        chart_df = pd.DataFrame({
            'Planning Classification': ['High Climate Hazard', 'Moderate Risk Zone', 'Optimal Safe Settlement'],
            'Percentage of Land Area': [high_hazard, moderate_risk, optimal_safe]
        })
        
        fig = px.bar(
            chart_df, 
            y='Planning Classification', 
            x='Percentage of Land Area', 
            orientation='h',
            color='Planning Classification',
            color_discrete_map={
                'High Climate Hazard': '#d73027',    # Strict Red
                'Moderate Risk Zone': '#fdae61',     # Strict Yellow
                'Optimal Safe Settlement': '#1a9850' # Strict Green
            }
        )
        fig.update_layout(showlegend=False, height=280, margin=dict(l=0, r=10, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Sampling data... Adjust the weight slider slightly to force a refresh.")

st.write("---")

# =================================================================================================================================
# 📋 THE PROFESSIONAL SUITABILITY REPORTING TABLE
# ===========================================================================================================================
st.markdown("### 📋 Zonal Statistics & Planning Evaluation Matrix")

if not df_metrics.empty:
    def label_zone(score):
        if score < 0.48: return "🔴 Critical Risk (Unsuitable)"
        elif score < 0.68: return "🟡 Moderate Hazard (Restricted)"
        else: return "🟢 Resilient Ground (Highly Suitable)"
        
    df_metrics['Planning Recommendation'] = df_metrics['Suitability Index'].apply(label_zone)
    df_metrics = df_metrics.sort_values(by='Suitability Index', ascending=False).reset_index(drop=True)
    
    st.dataframe(
        df_metrics,
        column_config={
            "Latitude": st.column_config.NumberColumn(format="%.4f"),
            "Longitude": st.column_config.NumberColumn(format="%.4f"),
            "Slope (Deg)": st.column_config.NumberColumn(format="%.1f°"),
            "Rain (mm/hr)": st.column_config.NumberColumn(format="%.3f mm/h"),
            "Suitability Index": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.3f")
        },
        use_container_width=True,
        hide_index=True
    )
    
    csv_data = df_metrics.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Spatial Assessment Matrix (CSV)",
        data=csv_data,
        file_name=f"uttarakhand_suitability_report_{end_date_str}.csv",
        mime="text/csv")
else:
    st.info("Generating spreadsheet matrix records...")