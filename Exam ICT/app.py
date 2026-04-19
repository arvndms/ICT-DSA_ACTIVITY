import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#Load saved objects
model = joblib.load('best_model.pkl')        # Random Forest classifier
le    = joblib.load('label_encoder.pkl')     
ohe   = joblib.load('one_hot_encoder.pkl')   


chart_data = pd.read_csv('chart_data.csv')

# sqft/bath min-max from training (for scaling) 
# These values come from  final_df before MinMaxScaler was applied
SQFT_MIN, SQFT_MAX = 5.0, 52272.0
BATH_MIN, BATH_MAX = 1.0, 40.0

# Locations
TOP_20 = list(le.classes_)   

# Area types from OHE 
AREA_TYPES = list(ohe.categories_[0])  

# Predict 
def predict(location, bhk, sqft, bath, balcony, area_type):
    
    loc = location if location in TOP_20 else 'Other'
    loc_encoded = le.transform([loc])[0]

   
    sqft_scaled = (sqft - SQFT_MIN) / (SQFT_MAX - SQFT_MIN)
    bath_scaled  = (bath - BATH_MIN) / (BATH_MAX - BATH_MIN)
    sqft_scaled  = float(np.clip(sqft_scaled, 0, 1))
    bath_scaled  = float(np.clip(bath_scaled, 0, 1))

    
    area_encoded = ohe.transform([[area_type]])
    area_cols    = ohe.get_feature_names_out(['area_type'])

    #  Build feature dataframe matching X_binary columns exactly
    base = pd.DataFrame([[sqft_scaled, bath_scaled, balcony, bhk, loc_encoded]],
                        columns=['total_sqft', 'bath', 'balcony', 'bhk', 'location_encoded'])
    area_df = pd.DataFrame(area_encoded, columns=area_cols)
    X = pd.concat([base, area_df], axis=1)

    # Predict
    pred  = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]  
    return pred, proba

# UI
st.title('🏠 Bengaluru House Price Predictor')
st.markdown('Predicts whether a property is **high value** (≥ 100 Lakhs) or not.')
st.markdown('---')

col1, col2 = st.columns(2)

with col1:
    location  = st.selectbox(' Location', sorted(TOP_20))
    bhk       = st.number_input(' BHK', min_value=1, max_value=10, value=2, step=1)
    sqft      = st.number_input(' Total Sqft', min_value=0.0, value=1200.0, step=50.0)

with col2:
    bath      = st.number_input(' Bathrooms', min_value=1, max_value=10, value=2, step=1)
    balcony   = st.number_input('Balconies', min_value=0, max_value=5, value=1, step=1)
    area_type = st.selectbox(' Area Type', AREA_TYPES)


errors = []
if sqft <= 0:
    errors.append('Total Sqft must be greater than 0.')
if sqft > 0 and (sqft / bhk) < 100:
    errors.append(f'Area too small for {bhk} BHK (min 100 sqft per BHK).')
if bath > bhk + 2:
    errors.append('Too many bathrooms for selected BHK.')

for e in errors:
    st.error(e)


if st.button(' Predict', disabled=bool(errors)):
    pred, proba = predict(location, bhk, sqft, bath, balcony, area_type)

    if pred == 1:
        st.success(f'✅ **High Value Property** (≥ 100 Lakhs) — Confidence: {proba*100:.1f}%')
    else:
        st.info(f'🏡 **Standard Property** (< 100 Lakhs) — Confidence: {(1-proba)*100:.1f}%')

st.markdown('---')

# Bar chart
st.subheader(f'📊 Top 5 Most Expensive Locations for {bhk} BHK')

filtered = chart_data[chart_data['bhk'] == bhk]

if len(filtered) == 0:
    st.warning(f'No data available for {bhk} BHK.')
else:
    top5 = filtered.groupby('location')['price'].mean().nlargest(5)

    fig, ax = plt.subplots(figsize=(8, 4))
    top5.plot(kind='bar', ax=ax, color='steelblue', edgecolor='none')
    ax.set_ylabel('Average Price (Lakhs)')
    ax.set_title(f'Top 5 Locations · {bhk} BHK')
    ax.axhline(y=100, color='red', linestyle='--', linewidth=1, label='High Value Threshold (100L)')
    ax.legend()
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
