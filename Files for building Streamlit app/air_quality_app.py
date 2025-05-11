import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from missingno import matrix as msno_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Set page configuration
st.set_page_config(
    page_title="Beijing Air Quality Analysis",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

warnings.filterwarnings('ignore')

# Load the dataset
@st.cache_data
def load_data():
    # Load the merged dataset
    try:
        df = pd.read_csv("merged_air_quality.csv")
        # Convert to datetime if needed
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error("Could not find the dataset file. Please ensure 'merged_air_quality.csv' is in the correct location.")
        return None

df = load_data()

# Page 1: General Data Information
def general_info():
    st.title("Beijing Air Quality Analysis")
    st.write("This dashboard provides insights into Beijing's air quality data from multiple monitoring stations.")
    
    st.subheader("Dataset Overview")
    st.write(df.head())
    
    st.subheader("Dataset Shape")
    st.write(f"The dataset contains {df.shape[0]} rows and {df.shape[1]} columns.")
    
    st.subheader("Data Types")
    st.write(df.dtypes)
    
    st.subheader("Missing Values Analysis")
    missing_values = df.isnull().sum().to_frame().rename(columns={0: 'Missing Values'})
    missing_values['% of Total Values'] = 100 * missing_values['Missing Values'] / len(df)
    st.dataframe(missing_values.style.background_gradient(cmap='Oranges'))
    
    # Missing values matrix visualization
    st.subheader("Missing Values Pattern")
    fig, ax = plt.subplots(figsize=(12, 6))
    msno_matrix(df, ax=ax)
    st.pyplot(fig)

# Page 2: EDA
def eda():
    st.title("Exploratory Data Analysis")
    st.write("Explore the air quality patterns and relationships between different pollutants.")
    
    # Basic statistics
    st.subheader("Summary Statistics")
    st.write(df.describe().T)
    
    # Site type distribution
    st.subheader("Data Distribution by Site Type")
    fig = px.pie(df, names='site_type', title='Proportion of Records by Site Type')
    st.plotly_chart(fig, use_container_width=True)
    
    # Time series analysis
    st.subheader("Pollutant Levels Over Time")
    
    # Create date column if not exists
    if 'Date' not in df.columns and all(col in df.columns for col in ['year', 'month', 'day']):
        df['Date'] = pd.to_datetime(df[['year', 'month', 'day']])
    
    pollutants = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    
    # Let user select pollutant and site type
    col1, col2 = st.columns(2)
    with col1:
        selected_pollutant = st.selectbox("Select Pollutant", pollutants)
    with col2:
        selected_site = st.selectbox("Select Site Type", ['All'] + list(df['site_type'].unique()))
    
    # Filter data based on selection
    if selected_site != 'All':
        filtered_df = df[df['site_type'] == selected_site]
    else:
        filtered_df = df.copy()
    
    # Resample to monthly averages
    monthly_df = filtered_df.set_index('Date').resample('M')[selected_pollutant].mean().reset_index()
    
    # Create plot
    fig = px.line(monthly_df, x='Date', y=selected_pollutant, 
                  title=f"Monthly Average {selected_pollutant} Levels ({selected_site})")
    st.plotly_chart(fig, use_container_width=True)
    
    # Correlation analysis
    st.subheader("Correlation Between Pollutants")
    
    # Calculate correlations
    corr_matrix = df[pollutants].corr()
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)
    
    # Boxplots by site type
    st.subheader("Pollutant Distribution by Site Type")
    selected_pollutant_box = st.selectbox("Select Pollutant for Boxplot", pollutants)
    
    fig = px.box(df, x='site_type', y=selected_pollutant_box, 
                 title=f"{selected_pollutant_box} Distribution by Site Type")
    st.plotly_chart(fig, use_container_width=True)

# Page 3: Model Building
def model_building():
    st.title("PM2.5 Prediction Model")
    st.write("Predict PM2.5 levels based on other air quality parameters.")
    
    # Prepare data
    features = ['PM10', 'SO2', 'NO2', 'CO', 'O3', 'TEMP', 'PRES', 'DEWP', 'WSPM']
    target = 'PM2.5'
    
    # Drop rows with missing target values
    model_df = df.dropna(subset=[target])
    
    # Impute missing feature values with median
    for col in features:
        model_df[col] = model_df[col].fillna(model_df[col].median())
    
    # Split data
    X = model_df[features]
    y = model_df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    st.subheader("Model Performance")
    st.write(f"Mean Squared Error: {mse:.2f}")
    st.write(f"R-squared: {r2:.2f}")
    
    # Feature importance
    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h', 
                 title='Feature Importance in PM2.5 Prediction')
    st.plotly_chart(fig, use_container_width=True)
    
    # Actual vs Predicted plot
    st.subheader("Actual vs Predicted PM2.5")
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred
    }).sample(100)  # Display subset for clarity
    
    fig = px.scatter(results_df, x='Actual', y='Predicted', 
                     title='Actual vs Predicted PM2.5 Values',
                     trendline="ols")
    fig.add_shape(type="line", line=dict(dash='dash'),
                  x0=min(y_test), y0=min(y_test),
                  x1=max(y_test), y1=max(y_test))
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction interface
    st.subheader("PM2.5 Prediction")
    st.write("Enter values to predict PM2.5 levels:")
    
    # Create input widgets for each feature
    inputs = {}
    cols = st.columns(3)
    for i, feature in enumerate(features):
        with cols[i % 3]:
            inputs[feature] = st.number_input(
                f"{feature}",
                min_value=float(df[feature].min()),
                max_value=float(df[feature].max()),
                value=float(df[feature].median()),
                step=0.1
            )
    
    if st.button("Predict PM2.5"):
        # Create input DataFrame
        input_df = pd.DataFrame([inputs])
        
        # Predict
        prediction = model.predict(input_df)
        
        st.success(f"Predicted PM2.5 level: {prediction[0]:.2f} µg/m³")

# Page 4: About
def about():
    st.title("About This Project")
    st.write("""
    ### Beijing Air Quality Analysis Dashboard
    
    This application provides insights into air quality data collected from multiple monitoring stations in Beijing, China.
    
    **Data Sources:**
    - Dongsi station (Urban)
    - Dingling station (Rural)
    - Changping station (Suburban)
    - Aotizhongxin station (Industrial)
    
    **Features:**
    - **General Data Information**: Overview of the dataset including structure, missing values, and data types.
    - **Exploratory Data Analysis**: Visualizations of pollutant trends, correlations, and distributions.
    - **Model Building**: Random Forest model to predict PM2.5 levels based on other air quality parameters.
    
    **Technical Details:**
    - Built with Python and Streamlit
    - Uses scikit-learn for machine learning
    - Visualizations with Matplotlib, Seaborn, and Plotly
    
    **Project by:** [Your Name]
    """)

# Main App
def main():
    # Sidebar Navigation
    st.sidebar.title("Navigation")
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Air_pollution_icon.svg/1200px-Air_pollution_icon.svg.png", 
                    width=100)
    
    page = st.sidebar.radio(
        "Go to",
        ["Home", "Data Overview", "Exploratory Analysis", "Prediction Model", "About"],
        index=0
    )
    
    # Display selected page
    if page == "Home":
        st.title("🌫️ Beijing Air Quality Analysis")
        st.write("""
        Welcome to the Beijing Air Quality Analysis dashboard. This interactive tool allows you to explore and analyze 
        air quality data collected from multiple monitoring stations across Beijing.
        
        Use the sidebar to navigate between different sections:
        - **Data Overview**: Basic information about the dataset
        - **Exploratory Analysis**: Visualizations and insights
        - **Prediction Model**: PM2.5 prediction tool
        - **About**: Project information
        """)
        
        if df is not None:
            st.subheader("Quick Stats")
            cols = st.columns(4)
            cols[0].metric("Total Records", len(df))
            cols[1].metric("Monitoring Stations", df['station'].nunique())
            cols[2].metric("Time Period", f"{df['year'].min()} - {df['year'].max()}")
            cols[3].metric("Primary Pollutant", "PM2.5")
            
            st.subheader("Sample Data")
            st.write(df.sample(5))
        
    elif page == "Data Overview" and df is not None:
        general_info()
    elif page == "Exploratory Analysis" and df is not None:
        eda()
    elif page == "Prediction Model" and df is not None:
        model_building()
    elif page == "About":
        about()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Beijing Air Quality Analysis**  
    [GitHub Repository](#) | [Data Source](#)
    """)

if __name__ == "__main__":
    if df is not None:
        main()
    else:
        st.error("Failed to load data. Please check the data file.")