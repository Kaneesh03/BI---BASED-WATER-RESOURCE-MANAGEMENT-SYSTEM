import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Water Management Dashboard", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .header-title {
        color: #1f77d2;
        font-size: 32px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 30px;
    }
</style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_data
def load_data():
    df = pd.read_csv('final_water_dataset.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    return df

df = load_data()

# Sidebar - Navigation and Filters
st.sidebar.markdown("## 🎯 Navigation & Filters")
page = st.sidebar.radio("Select Page", ["📊 Dashboard", "🔮 Predictions", "📈 Data Mining", "📉 Analytics"])

# Date range filter
date_range = st.sidebar.date_input(
    "Select Date Range",
    [df['Date'].min(), df['Date'].max()]
)

df_filtered = df[(df['Date'] >= pd.Timestamp(date_range[0])) & (df['Date'] <= pd.Timestamp(date_range[1]))]

# Zone filter
zones = st.sidebar.multiselect("Select Zones", df['Zone'].unique(), default=df['Zone'].unique())
df_filtered = df_filtered[df_filtered['Zone'].isin(zones)]

# Area Type filter
area_types = st.sidebar.multiselect("Select Area Types", df['Area_Type'].unique(), default=df['Area_Type'].unique())
df_filtered = df_filtered[df_filtered['Area_Type'].isin(area_types)]

# ============= DASHBOARD PAGE =============
if page == "📊 Dashboard":
    st.markdown('<div class="header-title">💧 Water Management Dashboard</div>', unsafe_allow_html=True)
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_supplied = df_filtered['Water_Supplied'].sum()
        st.metric("Total Water Supplied", f"{total_supplied:,.0f}", "units")
    
    with col2:
        total_consumed = df_filtered['Water_Consumed'].sum()
        st.metric("Total Water Consumed", f"{total_consumed:,.0f}", "units")
    
    with col3:
        avg_efficiency = (df_filtered['Water_Consumed'].sum() / df_filtered['Water_Supplied'].sum()) * 100
        st.metric("Efficiency Rate", f"{avg_efficiency:.1f}%", "Consumed/Supplied")
    
    with col4:
        total_population = df_filtered['Population'].max()
        st.metric("Population Served", f"{total_population:,}", "people")
    
    st.divider()
    
    # Graph 1: Water Supplied vs Consumed Over Time
    st.subheader("📊 Graph 1: Water Supply vs Consumption Trend")
    daily_data = df_filtered.groupby('Date').agg({
        'Water_Supplied': 'sum',
        'Water_Consumed': 'sum'
    }).reset_index()
    
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=daily_data['Date'], y=daily_data['Water_Supplied'],
                              mode='lines+markers', name='Water Supplied',
                              line=dict(color='#1f77d2', width=2)))
    fig1.add_trace(go.Scatter(x=daily_data['Date'], y=daily_data['Water_Consumed'],
                              mode='lines+markers', name='Water Consumed',
                              line=dict(color='#ff7f0e', width=2)))
    fig1.update_layout(title="Water Supplied vs Consumed Over Time",
                      xaxis_title="Date", yaxis_title="Water (units)",
                      hovermode='x unified', height=400)
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    # Graph 2: Water Consumption by Zone
    with col1:
        st.subheader("📊 Graph 2: Consumption by Zone")
        zone_data = df_filtered.groupby('Zone')['Water_Consumed'].sum().reset_index()
        fig2 = px.bar(zone_data, x='Zone', y='Water_Consumed',
                     color='Zone', title="Total Water Consumed by Zone",
                     labels={'Water_Consumed': 'Water Consumed (units)'})
        fig2.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Graph 3: Consumption by Area Type
    with col2:
        st.subheader("📊 Graph 3: Consumption by Area Type")
        area_data = df_filtered.groupby('Area_Type')['Water_Consumed'].sum().reset_index()
        fig3 = px.pie(area_data, values='Water_Consumed', names='Area_Type',
                     title="Water Consumption Distribution by Area Type")
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    # Graph 4: Population vs Water Consumption
    with col1:
        st.subheader("📊 Graph 4: Population vs Consumption")
        pop_data = df_filtered.groupby('Date').agg({
            'Population': 'max',
            'Water_Consumed': 'sum'
        }).reset_index()
        
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=pop_data['Date'], y=pop_data['Population'],
                                  mode='lines', name='Population',
                                  line=dict(color='#2ca02c', width=2)))
        fig4.add_trace(go.Scatter(x=pop_data['Date'], y=pop_data['Water_Consumed'],
                                  mode='lines', name='Water Consumed',
                                  yaxis='y2', line=dict(color='#d62728', width=2)))
        fig4.update_layout(title="Population vs Water Consumption",
                          xaxis_title="Date", yaxis_title="Population",
                          yaxis2=dict(title="Water Consumed (units)", overlaying='y', side='right'),
                          hovermode='x unified', height=400)
        st.plotly_chart(fig4, use_container_width=True)
    
    # Graph 5: Water Efficiency Ratio
    with col2:
        st.subheader("📊 Graph 5: Efficiency Ratio Over Time")
        daily_efficiency = df_filtered.groupby('Date').agg({
            'Water_Supplied': 'sum',
            'Water_Consumed': 'sum'
        }).reset_index()
        daily_efficiency['Efficiency'] = (daily_efficiency['Water_Consumed'] / 
                                         daily_efficiency['Water_Supplied']) * 100
        
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(x=daily_efficiency['Date'], y=daily_efficiency['Efficiency'],
                                  mode='lines+markers', name='Efficiency %',
                                  line=dict(color='#9467bd', width=2),
                                  fill='tozeroy'))
        fig5.update_layout(title="Water Efficiency Ratio (Consumed/Supplied)",
                          xaxis_title="Date", yaxis_title="Efficiency (%)",
                          hovermode='x unified', height=400)
        st.plotly_chart(fig5, use_container_width=True)

# ============= PREDICTIONS PAGE =============
elif page == "🔮 Predictions":
    st.markdown('<div class="header-title">🔮 Water Consumption Predictions</div>', unsafe_allow_html=True)
    
    # Prepare data for prediction
    pred_data = df.copy()
    pred_data['DayOfYear'] = pred_data['Date'].dt.dayofyear
    pred_data['Month'] = pred_data['Date'].dt.month
    pred_data['DayOfWeek'] = pred_data['Date'].dt.dayofweek
    pred_data = pd.get_dummies(pred_data, columns=['Zone', 'Area_Type'], drop_first=False)
    
    # Features and target
    feature_cols = [col for col in pred_data.columns if col not in ['Date', 'Water_Consumed']]
    X = pred_data[feature_cols]
    y = pred_data['Water_Consumed']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_test = model.predict(X_test)
    y_pred_all = model.predict(X)
    
    # Model Performance
    col1, col2, col3 = st.columns(3)
    with col1:
        mse = mean_squared_error(y_test, y_pred_test)
        st.metric("Mean Squared Error (Test)", f"{mse:.2f}")
    with col2:
        rmse = np.sqrt(mse)
        st.metric("RMSE (Test)", f"{rmse:.2f}")
    with col3:
        r2 = r2_score(y_test, y_pred_test)
        st.metric("R² Score (Test)", f"{r2:.4f}")
    
    st.divider()
    
    # Prediction visualization
    pred_df = pd.DataFrame({
        'Date': df['Date'],
        'Actual': y,
        'Predicted': y_pred_all
    })
    
    fig_pred = go.Figure()
    fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Actual'],
                                  mode='lines', name='Actual',
                                  line=dict(color='#1f77d2', width=2)))
    fig_pred.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted'],
                                  mode='lines', name='Predicted',
                                  line=dict(color='#ff7f0e', width=2, dash='dash')))
    fig_pred.update_layout(title="Actual vs Predicted Water Consumption",
                          xaxis_title="Date", yaxis_title="Water Consumed (units)",
                          hovermode='x unified', height=500)
    st.plotly_chart(fig_pred, use_container_width=True)
    
    # Feature importance
    st.subheader("🎯 Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False).head(10)
    
    fig_importance = px.bar(feature_importance, x='Importance', y='Feature',
                           orientation='h', title="Top 10 Important Features")
    fig_importance.update_layout(height=400)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Future prediction
    st.subheader("📅 Predict Future Consumption")
    days_ahead = st.slider("Days to predict ahead", 1, 30, 7)
    
    last_date = df['Date'].max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)
    
    # Create future data
    future_df = pd.DataFrame({'Date': future_dates})
    future_df['DayOfYear'] = future_df['Date'].dt.dayofyear
    future_df['Month'] = future_df['Date'].dt.month
    future_df['DayOfWeek'] = future_df['Date'].dt.dayofweek
    
    # Add zone and area type features (average)
    for col in [c for c in feature_cols if c.startswith('Zone_') or c.startswith('Area_Type_')]:
        future_df[col] = 1 / len([c for c in feature_cols if c.startswith('Zone_')])
    
    # Fill missing columns with 0
    for col in feature_cols:
        if col not in future_df.columns:
            future_df[col] = 0
    
    future_pred = model.predict(future_df[feature_cols])
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"Average predicted consumption for next {days_ahead} days: {future_pred.mean():.2f} units")
    with col2:
        st.warning(f"Max predicted consumption: {future_pred.max():.2f} units")

# ============= DATA MINING PAGE =============
elif page == "📈 Data Mining":
    st.markdown('<div class="header-title">📈 Data Mining & Analysis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Summary Statistics")
        st.dataframe(df_filtered[['Water_Supplied', 'Water_Consumed', 'Population']].describe().round(2))
    
    with col2:
        st.subheader("🔍 Zone Analysis")
        zone_stats = df_filtered.groupby('Zone').agg({
            'Water_Supplied': 'mean',
            'Water_Consumed': 'mean',
            'Population': 'max'
        }).round(2)
        st.dataframe(zone_stats)
    
    st.divider()
    
    # Detailed Data Table
    st.subheader("📋 Detailed Data View")
    show_rows = st.slider("Number of rows to display", 10, 100, 20)
    st.dataframe(df_filtered.head(show_rows), use_container_width=True)
    
    # Data exploration
    st.subheader("🔎 Data Exploration")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Columns in Dataset:**")
        st.write(df_filtered.columns.tolist())
    
    with col2:
        st.write("**Data Shape:**")
        st.write(f"Rows: {len(df_filtered)}, Columns: {len(df_filtered.columns)}")
    
    # Correlation heatmap
    st.subheader("🔗 Correlation Analysis")
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    corr_matrix = df_filtered[numeric_cols].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values,
                                         x=numeric_cols, y=numeric_cols,
                                         colorscale='RdBu', zmid=0))
    fig_corr.update_layout(title="Correlation Matrix", height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

# ============= ANALYTICS PAGE =============
elif page == "📉 Analytics":
    st.markdown('<div class="header-title">📉 Advanced Analytics</div>', unsafe_allow_html=True)
    
    # Water loss analysis
    st.subheader("💧 Water Loss Analysis")
    df_filtered['Water_Loss'] = df_filtered['Water_Supplied'] - df_filtered['Water_Consumed']
    df_filtered['Loss_Percentage'] = (df_filtered['Water_Loss'] / df_filtered['Water_Supplied']) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        total_loss = df_filtered['Water_Loss'].sum()
        st.metric("Total Water Loss", f"{total_loss:,.0f}", "units")
    
    with col2:
        avg_loss_pct = df_filtered['Loss_Percentage'].mean()
        st.metric("Avg Loss Percentage", f"{avg_loss_pct:.2f}%")
    
    with col3:
        max_loss_pct = df_filtered['Loss_Percentage'].max()
        st.metric("Max Loss Percentage", f"{max_loss_pct:.2f}%")
    
    st.divider()
    
    # Loss by zone
    st.subheader("Loss Distribution by Zone")
    loss_by_zone = df_filtered.groupby('Zone').agg({
        'Water_Loss': 'sum',
        'Loss_Percentage': 'mean'
    }).reset_index()
    
    fig_loss = px.bar(loss_by_zone, x='Zone', y='Water_Loss',
                     color='Loss_Percentage', title="Water Loss by Zone",
                     labels={'Water_Loss': 'Total Water Loss (units)'})
    st.plotly_chart(fig_loss, use_container_width=True)
    
    # Consumption per capita
    st.subheader("📊 Per Capita Consumption")
    df_filtered['Consumption_Per_Capita'] = (df_filtered['Water_Consumed'] / 
                                            df_filtered['Population']) * 1000
    
    fig_pc = px.box(df_filtered, x='Zone', y='Consumption_Per_Capita',
                   color='Area_Type', title="Per Capita Water Consumption")
    st.plotly_chart(fig_pc, use_container_width=True)
    
    # Anomaly detection
    st.subheader("🚨 Anomaly Detection")
    Q1 = df_filtered['Water_Consumed'].quantile(0.25)
    Q3 = df_filtered['Water_Consumed'].quantile(0.75)
    IQR = Q3 - Q1
    
    anomalies = df_filtered[(df_filtered['Water_Consumed'] < (Q1 - 1.5 * IQR)) | 
                           (df_filtered['Water_Consumed'] > (Q3 + 1.5 * IQR))]
    
    st.info(f"Found {len(anomalies)} anomalies in water consumption")
    if len(anomalies) > 0:
        st.dataframe(anomalies.head(10), use_container_width=True)
