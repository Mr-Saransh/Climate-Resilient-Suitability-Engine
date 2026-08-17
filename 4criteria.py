"""Climate-resilient urban suitability assessment dashboard.

Combines GPM precipitation, SRTM terrain slope, MODIS land-surface temperature
and OpenLandMap soil texture through a construction-focused AHP analysis.
"""

from datetime import date, timedelta
import json
from urllib.parse import quote
from urllib.request import Request, urlopen

import ee
import folium
from branca.element import MacroElement, Template
from folium.plugins import Fullscreen
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import folium_static


st.set_page_config(layout="wide", page_title="Climate Resilient Suitability Engine")

st.markdown(
    """
    <style>
    @keyframes map-loader-pulse {
        0%, 100% { opacity: .45; transform: scale(.96); }
        50% { opacity: 1; transform: scale(1); }
    }

    .map-loading {
        min-height: 510px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: .75rem;
        border: 1px solid rgba(49, 130, 189, .22);
        border-radius: .75rem;
        background: linear-gradient(135deg, #f7fbff, #eef7f2);
        color: #245b78;
    }

    .map-loading__orb {
        width: 2.75rem;
        height: 2.75rem;
        border: .28rem solid rgba(49, 130, 189, .18);
        border-top-color: #3182bd;
        border-radius: 50%;
        animation: map-loader-pulse 1s ease-in-out infinite;
    }

    /* streamlit-folium renders the completed map as a custom component. */
    div[data-testid="stCustomComponentV1"] iframe {
        border-radius: .75rem;
        overflow: hidden;
        box-shadow: 0 12px 30px rgba(21, 71, 52, .13);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

CRITERIA = ["Rainfall", "Slope", "Temperature", "Soil"]
RANDOM_INDEX = {3: 0.58, 4: 0.90}
SAATY_OPTIONS = {
    "1/9": 1 / 9, "1/8": 1 / 8, "1/7": 1 / 7, "1/6": 1 / 6,
    "1/5": 1 / 5, "1/4": 1 / 4, "1/3": 1 / 3, "1/2": 1 / 2,
    "1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0,
    "6": 6.0, "7": 7.0, "8": 8.0, "9": 9.0,
}
TEMPERATURE_PALETTES = {
    "Scientific (Blue → Red)": ["#2166ac", "#1a9850", "#fee08b", "#f46d43", "#d73027"],
    "Thermal": ["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"],
    "Warm": ["#ffffcc", "#fed976", "#fd8d3c", "#e31a1c", "#800026"],
}
SOIL_TEXTURE_PALETTE = [
    "#d5c36b", "#b96947", "#9d3706", "#ae868f", "#f86714", "#46d143",
    "#368f20", "#3e5a14", "#ffd557", "#fff72e", "#ff5a9d", "#ff005b",
]
USDA_TEXTURE_LABELS = {
    1: "Clay", 2: "Silty clay", 3: "Sandy clay", 4: "Clay loam",
    5: "Silty clay loam", 6: "Sandy clay loam", 7: "Loam",
    8: "Silt loam", 9: "Sandy loam", 10: "Silt", 11: "Loamy sand",
    12: "Sand",
}
# Screening scores for construction: clay-rich texture has shrink-swell risk;
# silt and loose sand need additional assessment. These are deliberately
# conservative and must not be used in place of a geotechnical investigation.
USDA_TEXTURE_CONSTRUCTION_SUITABILITY = {
    1: 0.15, 2: 0.20, 3: 0.30, 4: 0.60, 5: 0.55, 6: 0.70,
    7: 1.00, 8: 0.55, 9: 0.85, 10: 0.40, 11: 0.55, 12: 0.35,
}


class MapFlight(MacroElement):
    """Animate a Leaflet map from the previously viewed area to a new one."""

    _template = Template(
        """
        {% macro script(this, kwargs) %}
        (function () {
            const map = {{ this._parent.get_name() }};
            const destination = {{ this.destination | tojson }};
            const label = {{ this.label | tojson }};
            const targetZoom = {{ this.target_zoom | tojson }};
            const mapNode = map.getContainer();
            const style = document.createElement("style");
            style.textContent = `
                @keyframes climate-map-orbit {
                    to { transform: rotate(360deg); }
                }
                .climate-map-flight-overlay {
                    position: absolute;
                    inset: 0;
                    z-index: 1000;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    pointer-events: none;
                    background: linear-gradient(135deg, rgba(247, 251, 255, .94), rgba(238, 247, 242, .82));
                    opacity: 1;
                    transition: opacity .55s ease;
                }
                .climate-map-flight-card {
                    display: flex;
                    align-items: center;
                    gap: .7rem;
                    padding: .75rem 1rem;
                    border: 1px solid rgba(49, 130, 189, .22);
                    border-radius: 999px;
                    background: rgba(255, 255, 255, .92);
                    box-shadow: 0 8px 28px rgba(25, 74, 59, .14);
                    color: #245b78;
                    font: 600 14px/1.2 system-ui, sans-serif;
                }
                .climate-map-flight-orbit {
                    width: 1.1rem;
                    height: 1.1rem;
                    border: 3px solid rgba(49, 130, 189, .22);
                    border-top-color: #3182bd;
                    border-radius: 50%;
                    animation: climate-map-orbit .75s linear infinite;
                }
            `;
            document.head.appendChild(style);

            const overlay = document.createElement("div");
            overlay.className = "climate-map-flight-overlay";
            const card = document.createElement("div");
            card.className = "climate-map-flight-card";
            const orbit = document.createElement("div");
            orbit.className = "climate-map-flight-orbit";
            const message = document.createElement("span");
            message.textContent = `Flying to ${label}`;
            card.append(orbit, message);
            overlay.appendChild(card);
            mapNode.appendChild(overlay);

            const finishFlight = function () {
                overlay.style.opacity = "0";
                window.setTimeout(function () {
                    overlay.remove();
                    style.remove();
                }, 600);
            };

            map.whenReady(function () {
                window.setTimeout(function () {
                    map.invalidateSize();
                    overlay.style.opacity = ".48";
                    const flyToDestination = function () {
                        message.textContent = `Flying to ${label}`;
                        map.once("moveend", finishFlight);
                        map.flyTo(destination, targetZoom, {
                            animate: true,
                            duration: 1.8,
                            easeLinearity: 0.18,
                        });
                    };

                    // Pull back first so even nearby searches get a clear
                    // zoom-out -> travel -> zoom-in camera transition.
                    message.textContent = "Zooming out to plan the route";
                    map.once("moveend", flyToDestination);
                    map.flyTo(map.getCenter(), Math.min(map.getZoom(), 6), {
                        animate: true,
                        duration: 0.65,
                        easeLinearity: 0.22,
                    });
                }, 325);
            });
        }());
        {% endmacro %}
        """
    )

    def __init__(self, destination, label: str, target_zoom: int = 11):
        super().__init__()
        self._name = "MapFlight"
        self.destination = destination
        self.label = label
        self.target_zoom = target_zoom


def ahp_results(
    rain_slope: float,
    rain_temperature: float,
    rain_soil: float,
    slope_temperature: float,
    slope_soil: float,
    temperature_soil: float,
):
    """Build the construction-focused four-criterion AHP matrix."""
    matrix = np.array([
        [1.0, rain_slope, rain_temperature, rain_soil],
        [1.0 / rain_slope, 1.0, slope_temperature, slope_soil],
        [1.0 / rain_temperature, 1.0 / slope_temperature, 1.0, temperature_soil],
        [1.0 / rain_soil, 1.0 / slope_soil, 1.0 / temperature_soil, 1.0],
    ])
    normalized = matrix / matrix.sum(axis=0)
    weights = normalized.mean(axis=1)
    lambda_max = float(np.mean((matrix @ weights) / weights))
    ci = (lambda_max - len(CRITERIA)) / (len(CRITERIA) - 1)
    cr = ci / RANDOM_INDEX[len(CRITERIA)]
    return matrix, normalized, weights, lambda_max, ci, cr


def temperature_suitability(temperature_celsius):
    """Apply the requested piecewise MODIS LST suitability curve (0–1)."""
    return (ee.Image(1.0)
            .where(temperature_celsius.gt(20).And(temperature_celsius.lte(30)),
                   ee.Image(1.0).subtract(temperature_celsius.subtract(20).multiply(0.03)))
            .where(temperature_celsius.gt(30).And(temperature_celsius.lte(35)),
                   ee.Image(0.7).subtract(temperature_celsius.subtract(30).multiply(0.06)))
            .where(temperature_celsius.gt(35), 0)
            .clamp(0, 1)
            .rename("temperature_suitability"))


def soil_texture_suitability(texture_class):
    """Map USDA texture classes to a conservative construction screening score."""
    classes = list(USDA_TEXTURE_CONSTRUCTION_SUITABILITY)
    scores = [USDA_TEXTURE_CONSTRUCTION_SUITABILITY[texture] for texture in classes]
    return texture_class.remap(classes, scores, 0).rename("soil_suitability")


@st.cache_data(show_spinner=False, ttl=86400)
def geocode_area(area: str):
    """Return a Nominatim location's display name, bounds and map centre."""
    url = (
        "https://nominatim.openstreetmap.org/search?"
        f"q={quote(area)}&format=jsonv2&limit=1"
    )
    request = Request(url, headers={"User-Agent": "ClimateSuitabilityDashboard/1.0"})
    try:
        with urlopen(request, timeout=10) as response:
            matches = json.load(response)
    except Exception as exc:
        return None, f"Could not look up that area: {exc}"

    if not matches:
        return None, "No location was found. Try including the city, state, or country."

    match = matches[0]
    south, north, west, east = (float(value) for value in match["boundingbox"])
    return {
        "name": match["display_name"],
        "bounds": [west, south, east, north],
        "center": [(south + north) / 2, (west + east) / 2],
    }, None


def make_layers(
    start_date: str,
    end_date: str,
    weights: tuple[float, float, float, float],
    roi_bounds: tuple[float, float, float, float],
):
    """Create clipped and normalized Earth Engine layers for the MCDA overlay."""
    roi = ee.Geometry.Rectangle(list(roi_bounds))
    slope = ee.Terrain.slope(ee.Image("USGS/SRTMGL1_003").clip(roi)).rename("slope")
    rainfall = (ee.ImageCollection("NASA/GPM_L3/IMERG_V07").filterBounds(roi)
                .filterDate(start_date, end_date).select("precipitation").mean()
                .clip(roi).rename("rainfall"))
    # MOD11A2 LST_Day_1km is stored in Kelvin scaled by 0.02.
    temperature_celsius = (ee.ImageCollection("MODIS/061/MOD11A2").filterBounds(roi)
                           .filterDate(start_date, end_date).select("LST_Day_1km").mean()
                           .multiply(0.02).subtract(273.15).clip(roi)
                           .rename("temperature_celsius"))
    # OpenLandMap provides globally modelled USDA texture at 250 m. Foundation
    # screening uses 30 cm and 60 cm depths, rather than transient topsoil.
    soil_texture_product = ee.Image("OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02")
    soil_texture = soil_texture_product.select("b30").clip(roi).rename("soil_texture_class")
    deep_soil_texture = soil_texture_product.select("b60").clip(roi)
    clay_product = ee.Image("OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02")
    soil_clay_pct = (clay_product.select("b30").max(clay_product.select("b60"))
                     .clip(roi).rename("soil_clay_pct"))

    # Existing rainfall and slope normalisation methodology retained unchanged.
    rainfall_suitability = ee.Image(1.0).subtract(rainfall.divide(0.4)).clamp(0, 1).rename("rainfall_suitability")
    slope_suitability = ee.Image(1.0).subtract(slope.divide(25.0)).clamp(0, 1).rename("slope_suitability")
    temp_suitability = temperature_suitability(temperature_celsius)
    texture_score = soil_texture_suitability(soil_texture).min(
        soil_texture_suitability(deep_soil_texture)
    )
    clay_safety_cap = ee.Image(1.0).where(soil_clay_pct.gte(40), 0.25)
    soil_suitability = texture_score.min(clay_safety_cap).rename("soil_suitability")
    suitability = (rainfall_suitability.multiply(weights[0])
                   .add(slope_suitability.multiply(weights[1]))
                   .add(temp_suitability.multiply(weights[2]))
                   .add(soil_suitability.multiply(weights[3]))
                   .rename("suitability"))
    non_built_mask = ee.ImageCollection("ESA/WorldCover/v200").mosaic().select("Map").neq(50)
    return {
        "roi": roi, "rainfall": rainfall, "slope": slope,
        "temperature": temperature_celsius, "rainfall_suitability": rainfall_suitability,
        "slope_suitability": slope_suitability, "temperature_suitability": temp_suitability,
        "soil": soil_texture, "soil_clay": soil_clay_pct, "soil_suitability": soil_suitability,
        "suitability": suitability.updateMask(non_built_mask), "mask": non_built_mask,
    }


def tile_url(image, viz_params):
    return image.getMapId(viz_params)["tile_fetcher"].url_format


def geotiff_download_url(image, roi, scale: int, name: str) -> str:
    """Return an Earth Engine URL for one georeferenced GeoTIFF layer."""
    return image.getDownloadURL({
        "name": name,
        "region": roi,
        "scale": scale,
        "crs": "EPSG:4326",
        "format": "GEO_TIFF",
    })


@st.cache_data(show_spinner="Extracting spatial assessment samples…")
def extract_samples(
    start_date: str,
    end_date: str,
    weights: tuple[float, float, float, float],
    roi_bounds: tuple[float, float, float, float],
):
    """Create a compact table for charts, statistics and CSV export."""
    layers = make_layers(start_date, end_date, weights, roi_bounds)
    stack = (layers["suitability"].addBands(layers["slope"]).addBands(layers["rainfall"])
             .addBands(layers["temperature"]).addBands(layers["rainfall_suitability"])
             .addBands(layers["slope_suitability"]).addBands(layers["temperature_suitability"])
             .addBands(layers["soil"]).addBands(layers["soil_clay"])
             .addBands(layers["soil_suitability"])
             .updateMask(layers["mask"]))
    samples = stack.sample(region=layers["roi"], scale=3000, numPixels=135, geometries=True, seed=42)
    try:
        records = []
        for feature in samples.getInfo().get("features", []):
            props, coords = feature.get("properties", {}), feature["geometry"]["coordinates"]
            records.append({
                "Latitude": round(coords[1], 4), "Longitude": round(coords[0], 4),
                "Rainfall (mm/hr)": round(props.get("rainfall", 0), 3),
                "Slope (°)": round(props.get("slope", 0), 1),
                "Temperature (°C)": round(props.get("temperature_celsius", 0), 2),
                "Soil Texture (USDA, 30 cm)": USDA_TEXTURE_LABELS.get(
                    int(round(props.get("soil_texture_class", 0))), "Unknown"
                ),
                "Soil Clay (max, 30-60 cm, %)": round(props.get("soil_clay_pct", 0), 1),
                "Rainfall Suitability": round(props.get("rainfall_suitability", 0), 3),
                "Slope Suitability": round(props.get("slope_suitability", 0), 3),
                "Temperature Suitability": round(props.get("temperature_suitability", 0), 3),
                "Soil Suitability": round(props.get("soil_suitability", 0), 3),
                "Suitability Index": round(props.get("suitability", 0), 3),
            })
        return pd.DataFrame(records)
    except Exception:
        return pd.DataFrame()


def add_gee_layer(map_object, image, name, viz_params, opacity=0.75, enabled=True):
    """Add a named Earth Engine tile layer only when its toggle is enabled."""
    if enabled:
        folium.TileLayer(tiles=tile_url(image, viz_params), attr="Google Earth Engine", name=name,
                         overlay=True, opacity=opacity, show=True).add_to(map_object)


def classify(score: float) -> str:
    if score <= 0.40:
        return "High Risk"
    if score <= 0.68:
        return "Moderate"
    return "Suitable"


try:
    ee.Initialize(project="climate-resilition")
except Exception as exc:
    st.error(f"Google Earth Engine initialization failed: {exc}")
    st.stop()


st.title("Climate Resilient Suitability Engine")
st.caption("A four-criterion GIS decision-support system for climate-resilient construction screening")

if "study_area" not in st.session_state:
    default_area, default_error = geocode_area("Dehradun, Uttarakhand, India")
    if default_error:
        st.error(default_error)
        st.stop()
    st.session_state.study_area = default_area

area_changed = False
previous_map_center = None

with st.sidebar:
    st.header("Analysis controls")
    with st.expander("Study Area", expanded=True):
        with st.form("area_search"):
            area_query = st.text_input("Area, city, or address", value="Dehradun, Uttarakhand, India")
            search_area = st.form_submit_button("Use this area")
        if search_area:
            result, error = geocode_area(area_query.strip()) if area_query.strip() else (None, "Enter an area to search.")
            if error:
                st.error(error)
            else:
                previous_map_center = st.session_state.study_area["center"]
                st.session_state.study_area = result
                area_changed = previous_map_center != result["center"]
        selected_area = st.session_state.study_area
        west, south, east, north = selected_area["bounds"]
        st.caption(selected_area["name"])
        st.caption(f"{south:.4f}–{north:.4f}° N · {west:.4f}–{east:.4f}° E")
    with st.expander("Date Selection", expanded=True):
        end = st.date_input("End date", value=date.today())
        start = st.date_input("Start date", value=end - timedelta(days=30), max_value=end)
    with st.expander("Rainfall Controls"):
        show_rain = st.toggle("Rainfall layer", value=False)
        rain_opacity = st.slider("Rainfall opacity", 0.1, 1.0, 0.65, 0.05)
    with st.expander("Slope Controls"):
        show_slope = st.toggle("Slope layer", value=False)
        slope_opacity = st.slider("Slope opacity", 0.1, 1.0, 0.65, 0.05)
    with st.expander("Temperature Controls", expanded=True):
        show_temperature = st.toggle("Temperature layer", value=False)
        temperature_opacity = st.slider("Temperature opacity", 0.1, 1.0, 0.65, 0.05)
        temperature_palette_name = st.selectbox("Temperature palette", list(TEMPERATURE_PALETTES))
    with st.expander("Soil Controls", expanded=True):
        show_soil = st.toggle("Soil texture layer (30 cm)", value=False)
        soil_opacity = st.slider("Soil texture opacity", 0.1, 1.0, 0.70, 0.05)
        st.caption("OpenLandMap USDA texture at 250 m; the score also checks 60 cm texture and high clay.")
    with st.expander("AHP Settings — Analytical Hierarchy Process", expanded=True):
        st.caption("Select how strongly the first criterion is preferred to the second.")
        st.caption("Construction-first defaults rank soil, slope, rainfall, then temperature.")
        rain_slope_label = st.select_slider("Rainfall vs Slope", options=list(SAATY_OPTIONS), value="1/2")
        rain_temp_label = st.select_slider("Rainfall vs Temperature", options=list(SAATY_OPTIONS), value="2")
        rain_soil_label = st.select_slider("Rainfall vs Soil", options=list(SAATY_OPTIONS), value="1/4")
        slope_temp_label = st.select_slider("Slope vs Temperature", options=list(SAATY_OPTIONS), value="3")
        slope_soil_label = st.select_slider("Slope vs Soil", options=list(SAATY_OPTIONS), value="1/3")
        temp_soil_label = st.select_slider("Temperature vs Soil", options=list(SAATY_OPTIONS), value="1/6")
    with st.expander("Visualization", expanded=True):
        show_suitability = st.toggle("Suitability layer", value=True)
        suitability_opacity = st.slider("Suitability opacity", 0.1, 1.0, 0.78, 0.05)
    with st.expander("Export"):
        st.caption("Download the sampled spatial assessment table below.")
    with st.expander("Local GeoTIFF export", expanded=True):
        export_scale = st.select_slider(
            "Export resolution (metres)", options=[250, 500, 1000, 3000], value=1000,
            help=("1,000 m is recommended for this planning-level export. Some source datasets, "
                  "particularly rainfall, have a coarser native resolution."),
        )
        prepare_geotiffs = st.button(
            "Prepare GeoTIFF downloads",
            help=("Creates temporary download links for the current study area, date range, "
                  "and AHP weights."),
            use_container_width=True,
        )
        st.caption("Includes raw environmental layers, four criterion scores, and final suitability.")
        st.caption("For large study areas, use a coarser resolution; direct Earth Engine downloads are size-limited.")
        geotiff_export_results = st.empty()

matrix, normalized_matrix, weights, lambda_max, ci, cr = ahp_results(
    SAATY_OPTIONS[rain_slope_label],
    SAATY_OPTIONS[rain_temp_label],
    SAATY_OPTIONS[rain_soil_label],
    SAATY_OPTIONS[slope_temp_label],
    SAATY_OPTIONS[slope_soil_label],
    SAATY_OPTIONS[temp_soil_label],
)
weight_tuple = tuple(float(value) for value in weights)
start_str, end_str = start.isoformat(), (end + timedelta(days=1)).isoformat()
roi_bounds = tuple(float(value) for value in st.session_state.study_area["bounds"])
map_center = st.session_state.study_area["center"]
initial_map_center = previous_map_center if area_changed else map_center
initial_map_zoom = 10 if area_changed else 11

left, right = st.columns([3, 2])
with left:
    st.subheader("Geographic risk & suitability map")
    map_stage = st.empty()
    with map_stage.container():
        st.markdown(
            f"""
            <div class="map-loading">
                <div class="map-loading__orb"></div>
                <strong>{"Loading the new study area" if area_changed else "Preparing spatial layers"}</strong>
                <span>Fetching map tiles and suitability data…</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

layers = make_layers(start_str, end_str, weight_tuple, roi_bounds)
df_metrics = extract_samples(start_str, end_str, weight_tuple, roi_bounds)

export_context = json.dumps({
    "bounds": roi_bounds,
    "start": start_str,
    "end": end_str,
    "weights": weight_tuple,
    "scale": export_scale,
}, sort_keys=True)

if prepare_geotiffs:
    export_layers = {
        "Rainfall (GPM)": layers["rainfall"],
        "Slope (SRTM)": layers["slope"],
        "Temperature (MODIS LST)": layers["temperature"],
        "Soil texture (USDA)": layers["soil"],
        "Soil clay content": layers["soil_clay"],
        "Rainfall suitability": layers["rainfall_suitability"],
        "Slope suitability": layers["slope_suitability"],
        "Temperature suitability": layers["temperature_suitability"],
        "Soil suitability": layers["soil_suitability"],
        "Final suitability": layers["suitability"],
    }
    try:
        with st.spinner("Preparing GeoTIFF download links..."):
            st.session_state.geotiff_exports = {
                "context": export_context,
                "urls": {
                    label: geotiff_download_url(
                        image,
                        layers["roi"],
                        export_scale,
                        f"{label.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{end.isoformat()}",
                    )
                    for label, image in export_layers.items()
                },
            }
    except Exception as exc:
        st.session_state.pop("geotiff_exports", None)
        geotiff_export_results.error(f"GeoTIFF links could not be prepared: {exc}")

stored_exports = st.session_state.get("geotiff_exports")
if stored_exports and stored_exports["context"] == export_context:
    with geotiff_export_results.container():
        st.success("Your GeoTIFF layers are ready.")
        for label, url in stored_exports["urls"].items():
            st.link_button(f"Download {label}", url, use_container_width=True)
        metadata = {
            "study_area": st.session_state.study_area["name"],
            "bounds_west_south_east_north": roi_bounds,
            "date_range": {"start": start_str, "end_exclusive": end_str},
            "export_resolution_metres": export_scale,
            "ahp_weights": dict(zip(CRITERIA, weight_tuple)),
            "layers": list(stored_exports["urls"]),
        }
        st.download_button(
            "Download export metadata (JSON)",
            data=json.dumps(metadata, indent=2),
            file_name=f"suitability_export_metadata_{end.isoformat()}.json",
            mime="application/json",
            use_container_width=True,
        )
        st.caption("Links are temporary. Re-prepare them if a download link expires.")
elif stored_exports:
    geotiff_export_results.caption(
        "GeoTIFF settings changed. Prepare new download links for the current analysis."
    )

with left:
    with map_stage.container():
        fmap = folium.Map(
            location=initial_map_center,
            zoom_start=initial_map_zoom,
            tiles="OpenStreetMap",
            control_scale=True,
        )
        Fullscreen(position="topleft").add_to(fmap)
        add_gee_layer(fmap, layers["rainfall"], "Rainfall (GPM)", {"min": 0, "max": 1, "palette": ["#f7fbff", "#6baed6", "#08306b"]}, rain_opacity, show_rain)
        add_gee_layer(fmap, layers["slope"], "Slope (SRTM)", {"min": 0, "max": 35, "palette": ["#f7fcf5", "#74c476", "#00441b"]}, slope_opacity, show_slope)
        add_gee_layer(fmap, layers["temperature"], "Temperature (MODIS LST)", {"min": 15, "max": 45, "palette": TEMPERATURE_PALETTES[temperature_palette_name]}, temperature_opacity, show_temperature)
        add_gee_layer(fmap, layers["soil"], "Soil texture (USDA, 30 cm)", {"min": 1, "max": 12, "palette": SOIL_TEXTURE_PALETTE}, soil_opacity, show_soil)
        add_gee_layer(fmap, layers["suitability"], "Suitability index", {"min": 0, "max": 1, "palette": ["#d73027", "#fdae61", "#1a9850"]}, suitability_opacity, show_suitability)
        folium.LayerControl(collapsed=False).add_to(fmap)
        if area_changed:
            MapFlight(map_center, st.session_state.study_area["name"], target_zoom=11).add_to(fmap)
        folium_static(fmap, width=800, height=510)
    st.caption("Layers: GPM rainfall · SRTM slope · MODIS LST (blue/cold → red/hot) · suitability (red/risk → green/suitable)")

    st.caption("Soil criterion: OpenLandMap USDA texture at 30–60 cm, with a high-clay safety cap.")

with right:
    st.subheader("Construction-suitability AHP model")
    if cr < 0.10:
        st.success(f"Consistent pairwise judgements (CR = {cr:.3f})")
    else:
        st.error(f"Inconsistent Pairwise Judgements (CR = {cr:.3f})")
    st.dataframe(pd.DataFrame(matrix, index=CRITERIA, columns=CRITERIA).round(3), use_container_width=True)
    st.caption("Normalized AHP matrix")
    st.dataframe(pd.DataFrame(normalized_matrix, index=CRITERIA, columns=CRITERIA).round(3), use_container_width=True)
    weights_df = pd.DataFrame({"Criterion": CRITERIA, "Weight": weights}).set_index("Criterion")
    st.dataframe(weights_df.style.format("{:.3f}"), use_container_width=True)
    st.caption(f"Principal eigenvalue: {lambda_max:.3f} · CI: {ci:.3f} · CR: {cr:.3f}")

st.divider()
st.subheader("Statistics panel")
if df_metrics.empty:
    st.warning("No valid samples were returned for the selected date range. Try a broader range.")
else:
    metric_columns = {"Rainfall": "Rainfall (mm/hr)", "Slope": "Slope (°)", "Temperature": "Temperature (°C)", "Suitability": "Suitability Index"}
    metric_columns["Soil"] = "Soil Suitability"
    stat_columns = st.columns(len(metric_columns))
    for column, (label, field) in zip(stat_columns, metric_columns.items()):
        with column:
            st.markdown(f"**{label}**")
            st.caption(f"Mean: {df_metrics[field].mean():.2f}")
            st.caption(f"Min: {df_metrics[field].min():.2f}")
            st.caption(f"Max: {df_metrics[field].max():.2f}")

    st.divider()
    charts, distribution = st.columns([1, 1])
    df_metrics["Classification"] = df_metrics["Suitability Index"].apply(classify)
    with charts:
        st.subheader("Zonal distribution")
        chart_df = df_metrics["Classification"].value_counts().reindex(["High Risk", "Moderate", "Suitable"], fill_value=0).reset_index()
        chart_df.columns = ["Classification", "Samples"]
        fig = px.bar(chart_df, x="Classification", y="Samples", color="Classification",
                     color_discrete_map={"High Risk": "#d73027", "Moderate": "#fdae61", "Suitable": "#1a9850"})
        fig.update_layout(showlegend=False, height=280, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, use_container_width=True)
    with distribution:
        st.subheader("Suitability classification")
        for zone in ["High Risk", "Moderate", "Suitable"]:
            share = (df_metrics["Classification"] == zone).mean() * 100
            st.metric(zone, f"{share:.1f}%")

    st.divider()
    st.subheader("Zonal statistics & planning evaluation matrix")
    table = df_metrics.sort_values("Suitability Index", ascending=False).reset_index(drop=True)
    st.dataframe(table, use_container_width=True, hide_index=True,
                 column_config={"Suitability Index": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")})
    st.download_button("Download spatial assessment matrix (CSV)", table.to_csv(index=False).encode("utf-8"),
                       file_name=f"suitability_report_{end.isoformat()}.csv", mime="text/csv")

with st.expander("Methodology and data sources"):
    st.markdown("""
    This construction-screening assessment uses NASA GPM IMERG precipitation, SRTM elevation-derived slope, MODIS MOD11A2 daytime land-surface temperature, and OpenLandMap USDA soil texture. Soil uses the more conservative texture score at 30 cm and 60 cm, with a 0.25 safety cap where maximum clay content is at least 40%. The soil products are 250 m global digital soil maps, not direct satellite measurements.

    Construction-first AHP defaults rank soil, slope, rainfall, then temperature; the four-by-four matrix uses Saaty's random index of 0.90. The final score is a weighted linear combination of the four screening layers.

    Soil texture is only a planning-screening proxy. It cannot determine bearing capacity, shrink-swell behaviour, liquefaction, groundwater, bedrock depth, or foundation design. Any site selected through this dashboard requires a site-specific geotechnical investigation before construction.

    Sources: [OpenLandMap USDA Soil Texture Class](https://developers.google.com/earth-engine/datasets/catalog/OpenLandMap_SOL_SOL_TEXTURE-CLASS_USDA-TT_M_v02) and [OpenLandMap Clay Content](https://developers.google.com/earth-engine/datasets/catalog/OpenLandMap_SOL_SOL_CLAY-WFRACTION_USDA-3A1A1A_M_v02).
    """)
