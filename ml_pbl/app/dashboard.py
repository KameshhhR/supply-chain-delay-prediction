import sys
import os

# Auto-resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
from app.decision_system import SupplyChainDecisionSystem

st.set_page_config(
    page_title='Supply Chain Delay Intelligence',
    page_icon='|ud83d|ude9a',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title('\ud69a Supply Chain Delay Intelligence Dashboard')
st.caption('Explainable AI Decision Support System for Real-Time Shipment Management')

@st.cache_resource
def load_system():
    return SupplyChainDecisionSystem()

try:
    decision_sys = load_system()
except Exception as e:
    st.error('Error loading model: ' + str(e))
    st.stop()

st.sidebar.header('Shipment Configuration')
preset = st.sidebar.selectbox('Quick Preset Template:', ['Custom Order', 'Standard Domestic Order', 'Urgent Cross-Border Order', 'Bulk Wholesale Shipment'])

if preset == 'Standard Domestic Order':
    d_mode, d_mkt, d_reg = 'Standard Class', 'USCA', 'US Center'
    d_sales, d_qty, d_disc = 240.0, 2, 0.0
elif preset == 'Urgent Cross-Border Order':
    d_mode, d_mkt, d_reg = 'First Class', 'Pacific Asia', 'Southeast Asia'
    d_sales, d_qty, d_disc = 450.0, 3, 25.0
elif preset == 'Bulk Wholesale Shipment':
    d_mode, d_mkt, d_reg = 'Second Class', 'Europe', 'Western Europe'
    d_sales, d_qty, d_disc = 1200.0, 5, 80.0
else:
    d_mode, d_mkt, d_reg = 'Standard Class', 'Pacific Asia', 'Southeast Asia'
    d_sales, d_qty, d_disc = 327.75, 1, 13.11

ship_modes = ['Standard Class', 'Second Class', 'First Class', 'Same Day']
ship_mode = st.sidebar.selectbox('Shipping Mode', ship_modes, index=ship_modes.index(d_mode))
markets = ['Pacific Asia', 'USCA', 'Europe', 'LATAM', 'Africa']
market = st.sidebar.selectbox('Destination Market', markets, index=markets.index(d_mkt))

regions = ['Southeast Asia', 'South Asia', 'Eastern Asia', 'Western Europe', 'Central America', 'US Center', 'West of USA', 'South America']
region_idx = regions.index(d_reg) if d_reg in regions else 0
region = st.sidebar.selectbox('Order Region', regions, index=region_idx)
segment = st.sidebar.selectbox('Customer Segment', ['Consumer', 'Corporate', 'Home Office'])
category = st.sidebar.selectbox('Product Category', ['Sporting Goods', 'Cleats', 'Men Footwear', 'Women Apparel', 'Electronics', 'Cardio Equipment'])
dept = st.sidebar.selectbox('Department', ['Fitness', 'Apparel', 'Golf', 'Footwear', 'Fan Shop', 'Outdoors'])

st.sidebar.markdown('---')
sales = st.sidebar.slider('Gross Sales ($)', 10.0, 2000.0, float(d_sales), step=5.0)
quantity = st.sidebar.slider('Order Item Quantity', 1, 5, int(d_qty))
discount = st.sidebar.slider('Order Discount ($)', 0.0, 300.0, float(d_disc), step=1.0)
profit = st.sidebar.slider('Estimated Profit ($)', -200.0, 500.0, float(sales * 0.25), step=5.0)

sla_map = {'Same Day': 0, 'First Class': 1, 'Second Class': 2, 'Standard Class': 4}
scheduled_days = sla_map.get(ship_mode, 4)

order_payload = {
    'Order Id': 10429,
    'Shipping Mode': ship_mode,
    'Days for shipment (scheduled)': scheduled_days,
    'Market': market,
    'Order Region': region,
    'Customer Segment': segment,
    'Category Name': category,
    'Department Name': dept,
    'Sales': sales,
    'Order Item Quantity': quantity,
    'Order Item Discount': discount,
    'Benefit per order': profit,
    'Order Item Product Price': sales / quantity,
    'Customer Country': 'Puerto Rico',
    'Order Country': 'Indonesia' if market == 'Pacific Asia' else 'United States',
    'Customer State': 'PR',
    'Order State': 'Jakarta'
}

res = decision_sys.predict_shipment(order_payload)
prob = res['Probability']
risk_level = res['Risk Level']
conf = res['Confidence']
pred = res['Prediction']
thresh = res['Optimal Threshold Used']

tab1, tab2, tab3, tab4 = st.tabs(['\ufd2e Interactive Prediction', '\ufe90 Explainable AI (SHAP)', '\ufd4a 4-Model Benchmarks', '\ufd46 Batch Simulator'])

with tab1:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader('Delay Probability & Speedometer Gauge')
        fig = go.Figure(go.Indicator(
            mode='gauge+number+delta',
            value=prob * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': 'Delay Risk Score: ' + str(round(prob*100,1)) + '%', 'font': {'size': 22}},
            delta={'reference': thresh * 100, 'increasing': {'color': '#ff4b2b'}, 'decreasing': {'color': '#00b09b'}, 'prefix': 'vs Threshold: '},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': '#6b11ff', 'thickness': 0.35},
                'steps': [
                    {'range': [0, 45], 'color': 'rgba(0, 176, 155, 0.55)'},
                    {'range': [45, 65], 'color': 'rgba(247, 151, 30, 0.55)'},
                    {'range': [65, 80], 'color': 'rgba(255, 65, 108, 0.65)'},
                    {'range': [80, 100], 'color': 'rgba(138, 35, 135, 0.85)'}
                ],
                'threshold': {'line': {'color': '#00ffff', 'width': 5}, 'thickness': 0.85, 'value': thresh * 100}
            }
        ))
        fig.update_layout(height=330, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader('Decision Summary')
        if pred == 'DELAYED':
            st.error(f'\u26a0\ufe0f Outcome: {pred} ({risk_level} RISK')
        else:
            st.success(f'\u2705 Outcome: {pred} ({risk_level} RISK')
        
        st.metric('Model Confidence', conf)
        c1, c2 = st.columns(2)
        c1.metric('Scheduled SLA Promise', f'{scheduled_days} Days')
        c2.metric('Optimal Threshold', f'{thresh:.2f}')

        st.markdown('---')
        if prob >= thresh:
            st.warning(f'\u26a0\ufe0f Delay risk is **{prob*100:.1f}%** for **{ship_mode}** to **{region}**.\n\nActionable Mitigation: Prioritize warehouse picking and staging before 14:00 dispatch window.')
        else:
            st.success(f'\u2705 On-Time Guaranteed: Shipment is within SLA tolerance for **{ship_mode}** to **{region}** (Confidence: **{conf}**).')

with tab2:
    st.subheader('Explainable AI & SHAP Visualizations')
    c_s1, c_s2 = st.columns(2)
    with c_s1:
        if os.path.exists('outputs/figures/shap_feature_importance.png'):
            st.image('outputs/figures/shap_feature_importance.png', caption='Global Feature Importance (Mean |SHAP|)', use_container_width=True)
    with c_s2:
        if os.path.exists('outputs/figures/shap_summary_beeswarm.png'):
            st.image('outputs/figures/shap_summary_beeswarm.png', caption='SHAP Summary Beeswarm Plot', use_container_width=True)

with tab3:
    st.subheader('4-Model Benchmark Comparison Matrix')
    if os.path.exists('outputs/tables/model_comparison.csv'):
        df_comp = pd.read_csv('outputs/tables/model_comparison.csv')
        st.dataframe(df_comp, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists('outputs/figures/validation_roc_curves.png'):
            st.image('outputs/figures/validation_roc_curves.png', caption='Validation ROC Curves', use_container_width=True)
    with c2:
        if os.path.exists('outputs/figures/validation_confusion_matrices.png'):
            st.image('outputs/figures/validation_confusion_matrices.png', caption='Validation Confusion Matrices', use_container_width=True)

with tab4:
    st.subheader('Test Set Real-Time Shipment Batch Simulator')
    if st.button('Load 5 Random Real Test Shipments'):
        df_full = pd.read_csv('DataCoSupplyChainDataset.csv', encoding='latin1')
        samples = df_full.sample(5).to_dict(orient='records')
        sim_results = []
        for s in samples:
            out = decision_sys.predict_shipment(s)
            sim_results.append({
                'Order ID': out['Shipment ID'],
                'Shipping Mode': out['Shipping Mode'],
                'Region': out['Destination Region'],
                'Prediction': out['Prediction'],
                'Delay Risk': str(round(out['Probability']*100, 1)) + '%',
                'Confidence': out['Confidence'],
                'Risk Level': out['Risk Level']
            })
        st.dataframe(pd.DataFrame(sim_results), use_container_width=True)
