# Beijing Air Quality Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.3+-brightgreen)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-yellowgreen)

## 📌 Overview
This project analyzes air quality data from **4 monitoring stations** in Beijing (2013-2017) to:
- Identify pollution patterns and trends
- Explore relationships between pollutants and weather
- Build predictive models for PM2.5 levels

**Key Features**:
- Data merging from multiple sources
- Advanced missing value treatment
- Interactive visualizations
- Machine learning pipeline

## 📂 Dataset
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Beijing+Multi-Site+Air-Quality+Data)

**Stations**:
1. Dongsi (Urban)
2. Dingling (Rural)  
3. Changping (Suburban)
4. Aotizhongxin (Industrial)

**Variables**:
- Pollutants: PM2.5, PM10, SO₂, NO₂, CO, O₃
- Weather: Temperature, Pressure, Humidity, Wind
- Temporal: Year, Month, Day, Hour

## 🛠️ Implementation
### 1. Data Processing
```python
# Handle missing values
df['PM2.5'].fillna(df.groupby('station')['PM2.5'].transform('median'), inplace=True)

# Feature engineering
df['pollution_ratio'] = df['PM2.5'] / df['PM10']  # Fine/coarse particle ratio
