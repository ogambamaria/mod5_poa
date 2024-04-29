import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report

st.set_page_config(page_title='CLM Prediction App', layout='wide')
st.title('CLM Prediction App')

# Sidebar and additional info
st.sidebar.header("About the App")
st.sidebar.info("This Streamlit app is designed to load, preprocess, visualize, and model data using CatBoost with tuned parameters for prediction.")

# Load your model
st.cache_data()
def load_model():
    return joblib.load('catboost_model_clm.pkl')

model = load_model()

# Load your data
st.cache_data()
def load_data():
    data = pd.read_csv('data/saved_data.csv', low_memory=False, index_col=0)
    # Preprocessing steps if needed
    return data

input_df = load_data()

st.markdown("""
Want to view other ways the model has been used? Check out the <a href="https://app.powerbi.com/view?r=eyJrIjoiZWEwNjIxZWYtYTFjNS00Y2U5LTk3MDQtZDczNzIxMDRjODJiIiwidCI6ImE0NjIyOWM3LTIxZDEtNDE3ZC1hMWNiLTE4NTdhMDdkMjc2NSIsImMiOjh9&pageName=ReportSection5f8ea718003bde305590" target="_blank">dashboard</a> for more details.
""", unsafe_allow_html=True)
st.markdown("""
See the data preparation and model development steps: <a href="https://github.com/ogambamaria/mod5_poa" target="_blank">Jupyter Notebook</a>.
""", unsafe_allow_html=True)

predictions = model.predict(input_df)
input_df['Predictions'] = predictions

# Model performance (assuming actual values are present)
report = classification_report(input_df['ServiceSatisfaction'], predictions, output_dict=True)
st.write('Classification Report:', pd.DataFrame(report).transpose())

# Feature Importance Chart
feature_importances = pd.Series(model.feature_importances_, index=model.feature_names_)
fig = px.bar(feature_importances, x=feature_importances.values, y=feature_importances.index, orientation='h')
st.plotly_chart(fig)