import sys
import os
import streamlit as st
import importlib
import pandas as pd
import numpy as np
import pydeck as pdk
import torch
import json
import pickle
import h5py
import datetime
from torch_geometric.data import Data
from pandas.tseries.holiday import USFederalHolidayCalendar


current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, "../../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

MODEL_REGISTRY = {
    "GCN Baseline": ("src.models.gcn_baseline", "GCNBaseline", "run_20260421_222234/best_model.pt"),
    "GCN Attention": ("src.models.gcn_attention", "GCNSpatioTemporalAttention", "run_20260421_222540/best_model.pt"),
    "SpatioTemporal GNN": ("src.models.stgcn", "SpatioTemporalGNN", "run_20260421_230619/best_model.pt"),
    "GCN Transformer": ("src.models.gcn_transformer", "SpatioTemporalTransformer", "run_20260421_233428/best_model.pt")
}


def get_real_metrics(selection):
    try:
        _, _, ckpt_rel_path = MODEL_REGISTRY[selection]
        metrics_path = os.path.join(root_dir, "ckpts", os.path.dirname(ckpt_rel_path), "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                data = json.load(f)
                return {"MAE": round(float(data.get("mae", 0.0)), 4), 
                        "RMSE": round(float(data.get("rmse", 0.0)), 4),
                        "MSE": round(float(data.get("mse", 0.0)), 4),
                        "Best Val Loss": round(float(data.get("best_val_loss", 0.0)), 6)
                }
    except: pass
    # return {"MAE": 0.0, "RMSE": 0.0}
    return {"MAE": 0.0, "RMSE": 0.0, "MSE": 0.0, "Best Val Loss": 0.0}



@st.cache_data
def load_dataset_resources(city_choice):
    sub_path, h5_file, pkl_file = ("datasets/raw/metr_la", "metr_la.h5", "adj_metr_la.pkl") if "LA" in city_choice else ("datasets/raw/pems-bay", "pems_bay.h5", "adj_mx_bay.pkl")
    start_date = "2012-03-01 00:00:00" if "LA" in city_choice else "2017-01-01 00:00:00"
    
    h5_path = os.path.abspath(os.path.join(root_dir, sub_path, h5_file))
    pkl_path = os.path.abspath(os.path.join(root_dir, sub_path, pkl_file))
    
    with h5py.File(h5_path, 'r') as f:
        data_key = 'df/block0_values' if 'df' in f else list(f.keys())[0]
        raw_data = f[data_key][:]
        idx = pd.date_range(start=start_date, periods=len(raw_data), freq='5min')
        traffic_df = pd.DataFrame(raw_data, index=idx)

    with open(pkl_path, 'rb') as f:
        pkl_content = pickle.load(f, encoding='latin1')
        adj_mx = pkl_content[2] if len(pkl_content) > 2 else pkl_content[1]
    
    num_nodes = adj_mx.shape[0]
    edge_index = torch.from_numpy(adj_mx).to_sparse().indices()
    lats = np.linspace(34.00, 34.15, num_nodes) + np.random.normal(0, 0.005, num_nodes)
    lons = np.linspace(-118.35, -118.15, num_nodes) + np.random.normal(0, 0.005, num_nodes)

    return traffic_df, edge_index, num_nodes, lats, lons

# UI SETUP
st.set_page_config(page_title="Traffic Lab", layout="wide")

city_choice = "Los Angeles (METR-LA)"

st.sidebar.title("City: Los Angeles")
st.sidebar.caption("Dataset: METR-LA")
st.sidebar.markdown("---")

model_step = st.sidebar.selectbox("Model Architecture", options=list(MODEL_REGISTRY.keys()))

traffic_df, REAL_EDGE_INDEX, NODE_COUNT, lats, lons = load_dataset_resources(city_choice)
all_model_metrics = {name: get_real_metrics(name) for name in MODEL_REGISTRY.keys()}

min_ts = traffic_df.index.min()
max_ts = traffic_df.index.max()

d = st.sidebar.date_input(
    "Date", 
    value=min_ts.date(),         
    min_value=min_ts.date(),
    max_value=max_ts.date()      
)

time_options = [datetime.time(hour=h, minute=m) for h in range(24) for m in (0, 15, 30, 45)]
time_labels = [t.strftime("%I:%M %p") for t in time_options]

selected_label = st.sidebar.selectbox("Time", options=time_labels, index=32)
t = time_options[time_labels.index(selected_label)]

idx_val = max(12, traffic_df.index.get_indexer([pd.Timestamp.combine(d, t)], method='nearest')[0])


@st.cache_resource
def load_model_instance(selection):
    module_path, class_name, ckpt_rel_path = MODEL_REGISTRY[selection]
    ckpt_path = os.path.join(root_dir, "ckpts", ckpt_rel_path)
    module = importlib.import_module(module_path)
    model = getattr(module, class_name)(in_feats=12, hidden_feats=64, out_feats=12)
    if os.path.exists(ckpt_path): model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    return model.eval()


active_model = load_model_instance(model_step)

def get_predictions(model_obj, start_idx):
    x_input = torch.tensor(traffic_df.iloc[start_idx-12 : start_idx, :].values.T).to(torch.float32)
    with torch.no_grad():
        try:
            pred = model_obj(x_input.unsqueeze(0), REAL_EDGE_INDEX).cpu().numpy().squeeze()
            
        except: 
            return x_input[:, -1].numpy()

speeds = get_predictions(active_model, idx_val)
df_map = pd.DataFrame({'lat': lats, 'lon': lons, 'speed': speeds, 'weight': 70 - speeds})


# THE MAP 
glow_gradient = [
    [0, 255, 100, 80], 
    [255, 255, 0, 160],  
    [255, 120, 0, 220], 
    [255, 0, 0, 255]   
]

heatmap = pdk.Layer(
    "HeatmapLayer",
    data=df_map,
    get_position="[lon, lat]",
    get_weight="weight",
    color_range=glow_gradient,
    threshold=0.03,
    radius_pixels=90,
    intensity=3
)

scatterplot = pdk.Layer(
    "ScatterplotLayer",
    data=df_map,
    get_position="[lon, lat]",
    get_fill_color="speed < 30 ? [255, 0, 0, 255] : (speed < 50 ? [255, 255, 0, 255] : [0, 255, 100, 200])",
    get_radius=150,
    stroked=True,          
    get_line_color=[0, 0, 0],  
    get_line_width=30,         
    pickable=True
)

st.pydeck_chart(pdk.Deck(
    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    initial_view_state=pdk.ViewState(latitude=lats.mean(), longitude=lons.mean(), zoom=11.2, pitch=45),
    layers=[heatmap, scatterplot],
    tooltip={"text": "Predicted Speed: {speed:.1f} mph"}
))



# PERFORMANCE 
st.subheader("📊 Performance Comparison")
curr_mae = all_model_metrics[model_step]["MAE"]

if "prev_model" not in st.session_state: 
    st.session_state.prev_model = "GCN Baseline"

prev_mae = all_model_metrics[st.session_state.prev_model]["MAE"]

c1, c2 = st.columns(2)
with c1: 
    st.info(f"**Model:** {model_step}  \n**Time:** {traffic_df.index[idx_val].strftime('%I:%M %p')}")

with c2:
    gain = ((prev_mae - curr_mae) / prev_mae * 100) if prev_mae > 0 else 0
    
    st.metric(
        label="Accuracy Change", 
        value=f"{gain:.1f}%", 
        delta=f"{gain:.1f}% vs {st.session_state.prev_model}",
        delta_color="normal" 
    )


st.session_state.prev_model = model_step


table_rows = []
for name, m in all_model_metrics.items():
    table_rows.append({
        "Model": name,
        "MAE": m.get("MAE", 0.0),
        "RMSE": m.get("RMSE", 0.0),
        "MSE": m.get("MSE", 0.0), 
        "Best Val Loss": m.get("Best Val Loss", 0.0) 
    })

perf_df = pd.DataFrame(table_rows)

st.dataframe(
    perf_df, 
    use_container_width=True, 
    hide_index=True,
    column_config={
        "MAE": st.column_config.NumberColumn(format="%.4f"),
        "RMSE": st.column_config.NumberColumn(format="%.4f"),
        "MSE": st.column_config.NumberColumn(format="%.4f"),
        "Best Val Loss": st.column_config.NumberColumn(format="%.6f"),
    }
)
