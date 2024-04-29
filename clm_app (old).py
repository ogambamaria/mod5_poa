import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title='CLM Prediction App', layout='wide')
st.title('CLM Prediction App')

# Sidebar and additional info
st.sidebar.header("About the App")
st.sidebar.info("This Streamlit app is designed to load, preprocess, visualize, and model data using CatBoost with tuned parameters for prediction.")
st.sidebar.markdown("""
Want to view other ways the model has been used? Check out the <a href="https://app.powerbi.com/view?r=eyJrIjoiZWEwNjIxZWYtYTFjNS00Y2U5LTk3MDQtZDczNzIxMDRjODJiIiwidCI6ImE0NjIyOWM3LTIxZDEtNDE3ZC1hMWNiLTE4NTdhMDdkMjc2NSIsImMiOjh9&pageName=ReportSection5f8ea718003bde305590" target="_blank">dashboard</a> for more details.
""", unsafe_allow_html=True)

# Initialize the session state
if 'use_default_data' not in st.session_state:
    st.session_state['use_default_data'] = True

# Load the pre-trained model
def load_model():
    return joblib.load('catboost_model_clm.pkl')

model = load_model()

# Define how to load and clean data
def load_data():
    if st.session_state['use_default_data']:
        return pd.read_csv('data/imonitor_1703.csv', low_memory=False)
    else:
        return pd.read_csv(uploaded_file)

uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"], on_change=lambda: setattr(st.session_state, 'use_default_data', False))

if st.button('Use Original Dataset'):
    st.session_state['use_default_data'] = True
    st.experimental_rerun()

df_input = load_data() if uploaded_file or st.session_state['use_default_data'] else pd.DataFrame()

def clean_and_preprocess_data(df):
    # Standard preprocessing steps
    cols_to_drop = [col for col in df.columns if "Please specify" in col or "Please Specify" in col]
    df.drop(cols_to_drop, axis=1, inplace=True)
    df.drop_duplicates(inplace=True)  # Drop duplicates
    df.columns = df.columns.map(lambda x: x.strip())

    # Drop specific columns that are not needed
    unnecessary_columns = [
        "Survey ID",
        "Facility name and MFL Code if applicable",
        "What is your month; and year of birth",
        "How do you consider yourself?",
        "What is the highest level of education you completed?",
        "What is your current marital status?",
        "Which county do you currently live in?",
        "What are your sources of income?",
        "Facility name",
        "What did you like about the services you received?",
        "What did you not like about the services you received?",
        "In your opinion what would you like to be improved?",
        "In your opinion what can be done to improve access to the services you seek at the facility?",
        "Facility name denied service",
        "Why",
        "Were reasons provided as to why these services were not available?",
        "Were reasons provided as to why these services were not available?.1",
        "What are the barriers to uptake of VMMC by males 25+years and above?",
        "What are some of the current site level practices that community members like and would love to maintain for KP/PP ?",
        "What would you like this facility to change/do better?",
        "Throughout your visit what did you find interesting/pleasing about this facility that should be emulated by other facilities?",
        "What do you think can be improved",
        "Anything else that you would like to mention?",
        "What are the top 1-3 things you like about this facility with regards to care and treatment?",
        "What are the top 1-3 things you don’t like about this facility with regards to care and treatment?",
        "how long do you wait on average to get a service; which service was that?",
        "how long do you wait on average to get your lab test result?",
        "Specify the support group you belong to"
    ]
    columns_to_drop = [col for col in unnecessary_columns if col in df.columns]
    df.drop(columns=columns_to_drop, axis=1, inplace=True)

    # Rename columns
    column_name_mapping = {
        "Created Date": "Date",
        "Organization name coordinating the feedback from the clients": "OrgFeedbackCoordinator",
        "Facility ownership": "FacilityOwnership",
        "County": "FacilityCounty",
        "For how long have you been accessing services (based on the expected package of services) in this facility?": "ServiceAccessDuration",
        "Are you aware of the package of services that you are entitled to?": "ServicesAwareness",
        "According to you; which HIV related services are you likely to receive in this facility?": "ExpectedHIVServices",
        "Is there a service that you needed that was not provided?": "UnprovidedService",
        "Facility name no service": "UnprovidedServiceFacilityName",
        "For that service that was not provided; were you referred?": "ReferralForUnprovidedService",
        "If referred; did you receive the service where you were referred to?": "ReferralServiceReceived",
        "If Yes which Service/Test/Medicine": "ReceivedServiceDetail",
        "On a scale of 1 to 5; how satisfied are you with the package of services received in this facility? If 1 is VERY UNSATISFIED and 5 is VERY SATISFIED.": "ServiceSatisfaction",
        "Do you face any challenges when accessing the services at the facility?": "AccessChallenges",
        "Common issues that can be added in the drop-down box": "CommonIssuesDropdown",
        "Was confidentiality considered while you were being served?": "Confidentiality",
        "Are there age-appropriate health services for specific groups?": "AgeAppropriateServices",
        "Does the facility allow you to share your concerns with the administration?": "ConcernsSharing",
        "Do you know your health-related rights as a client of this facility?": "RightsAwareness",
        "Have you ever been denied services at this facility?": "ServiceDenial",
        "Are you comfortable with getting services at this facility": "ComfortWithServices",
        "Have you ever been counseled?": "CounselingReceived",
        "Did you identify any gaps in the facility when you tried to access the services": "IdentifiedGaps",
        "Service type": "ServiceGapsType",
        "Are the HIV testing services readily available when required?": "HIVTestingAvailability",
        "Have you ever Interrupted your treatment?": "TreatmentInterruption",
        "Are the PMTCT services readily available when required?": "PMTCTServiceAvailability",
        "Are the HIV prevention; testing; treatment and care services adequate for KPs?": "KPServiceAdequacy",
        "Facility Level": "FacilityLevel",
        "Facility Operation times": "OperationTimes",
        "Facility Operation Days": "OperationDays",
        "What are your preferred days of visiting the facility": "PreferredVisitDays",
        "What are your preferred time of visiting the facility": "PreferredVisitTimes",
        "On a scale of 1-5; how clean do you find the facility?": "FacilityCleanliness",
        "How do you reach this facility?": "FacilityAccessMode",
        "How long does it take to reach this facility?": "FacilityAccessTime",
        "On a scale of 1-5; how accessible do you find this facility?": "FacilityAccessibility",
        "Do you consider the waiting time to be seen at this facility long?": "GeneralWaitingTime",
        "Do you consider the waiting time for lab test results long?": "LabResultsWaitingTime",
        "Does the facility offer support groups?": "SupportGroupAvailability",
        "In your opinion are the services offered at this facility youth friendly?": "YouthFriendlyServices",
        "What measures have been put in place to create GBV awareness and its harmful effects within the community?": "GBVAwarenessMeasures",
        "PWD In your opinion are the services offered at this facility persons-with-disability friendly?": "PWDFriendlyServicesOpinion"
    }
    df.rename(columns={k: v for k, v in column_name_mapping.items() if k in df.columns}, inplace=True)

    # Clean specific data fields
    df = standardize_facility(df, 'FacilityLevel')
    df = replace_symbols_and_words1(df, 'FacilityAccessTime')
    df = replace_symbols_and_words2(df, 'FacilityAccessTime')
    df = replace_county(df, 'FacilityCounty')
    df = replace_dont_know(df, 'GeneralWaitingTime')
    df = replace_mixed_with_text(df, 'FacilityCleanliness')
    df = standardize_gbv_awareness(df, 'GBVAwarenessMeasures')

    return df

def standardize_facility(df, column_name):
    # Mapping for consolidating variations of satisfaction levels
    satisfaction_map = {
        1.0: 'Community Health Unit',
        2.0: 'Dispensaries and Private Clinics',
        3.0: 'Health Centers',
        4.0: 'Sub-County Hospitals',
        5.0: 'County Referral Hospitals',
        6.0: 'National Referral Hospitals',
    }
    
    # Replace values based on the map
    df[column_name] = df[column_name].replace(satisfaction_map)
    return df

def replace_symbols_and_words1(df, column_name):
    df[column_name] = df[column_name].str.replace('<', 'Less than', regex=False)
    df[column_name] = df[column_name].str.replace('>', 'More than', regex=False)
    df[column_name] = df[column_name].str.replace('minutes', 'mins', regex=False)
    return df

def replace_symbols_and_words2(df, column_name):
    df[column_name] = df[column_name].str.replace('Less than 30mins', 'Less than 30 mins', regex=False)
    df[column_name] = df[column_name].str.replace('More than45 mins', 'More than 45 mins', regex=False)
    return df

def replace_county(df, column_name):
    df[column_name] = df[column_name].str.replace('Homabay', 'Homa Bay', regex=False)
    return df

def convert_mixed_dates(date_column):
    # Define the epoch start for Excel's serial date format
    excel_epoch = pd.Timestamp('1899-12-30')
    converted_dates = []

    for date in date_column:
        if isinstance(date, str) and re.match(r'^\d+(\.\d+)?$', date):
            # If it's a string that looks like a serial date, convert it
            serial_value = float(date)
            converted_date = excel_epoch + pd.to_timedelta(serial_value, unit='D')
        elif isinstance(date, (int, float)):
            # If it's a numeric type, assume it's a serial date
            converted_date = excel_epoch + pd.to_timedelta(date, unit='D')
        else:
            # Otherwise, try to parse it as a regular date
            converted_date = pd.to_datetime(date, errors='coerce')

        # Append the result, which will be NaT (Not a Time) if parsing failed
        converted_dates.append(converted_date)

    return pd.Series(converted_dates)

def replace_dont_know(df, column):
    df[column] = df[column].replace("Dont Know", "Do not know", regex=False)
    return df

def replace_mixed_with_text(df, column):
    satisfaction_map = {1: 'Very Unsatisfied', 1: 'Dissatisfied', 2: 'Unsatisfied', 3: 'Okay', 4: 'Satisfied', 5: 'Very Satisfied'}
    df[column] = df[column].apply(lambda x: satisfaction_map.get(int(x[0]), x) if isinstance(x, str) and x.isdigit() else x)
    return df

def standardize_gbv_awareness(df, column):
    df[column] = df[column].replace({
        'Is there a desk to report GBV as community or individual': 'Presence of GBV Desk',
        'Are there training events on GBV for the community': 'Community trained on GBV'
    }, regex=False)
    return df

# Clean and preprocess the data
df_cleaned = clean_and_preprocess_data(df_input) if not df_input.empty else pd.DataFrame()

# Visualization of cleaned data
st.header("Data Visualization")
feature_to_plot = st.selectbox("Select a Feature to Plot", df_cleaned.columns)
if st.button("Generate Plot"):
    fig = px.histogram(df_cleaned, x=feature_to_plot)
    st.plotly_chart(fig)

def feature_engineering(df):
    # Check for 'ServiceSatisfaction' before starting feature engineering
    if 'ServiceSatisfaction' not in df.columns:
        raise ValueError("ServiceSatisfaction column is missing from the dataframe.")

    # Define columns to encode and perform encoding
    columns_to_encode = ['ExpectedHIVServices', 'OperationTimes', 'OperationDays', 'PreferredVisitDays', 'PreferredVisitTimes', 'GBVAwarenessMeasures']
    for col in columns_to_encode:
        split_series = df[col].str.replace(' ', '').str.split(';')
        encoded = split_series.str.join('|').str.get_dummies()
        encoded.columns = [f"{col}_{option}" for option in encoded.columns]
        df = df.join(encoded)
    df.drop(columns=columns_to_encode, axis=1, inplace=True)

    # Remove columns with a high percentage of missing values
    missing_percentage = df.isnull().mean() * 100
    threshold = 60
    columns_to_drop = missing_percentage[missing_percentage > threshold].index.tolist()
    df.drop(columns=columns_to_drop, axis=1, inplace=True)

    # Drop rows with too many missing values
    threshold_percentage = 100
    threshold = len(df.columns) * (threshold_percentage / 100)
    data = df.dropna(thresh=threshold).copy()

    # Ensure 'ServiceSatisfaction' is still in the dataframe
    if 'ServiceSatisfaction' not in data.columns:
        raise ValueError("ServiceSatisfaction column was dropped during processing.")

    # Define ordinal and nominal variables correctly
    nominal_vars = [col for col in data.columns if data[col].dtype == 'object' and col != 'ServiceSatisfaction']
    encoded_data = pd.get_dummies(data, columns=nominal_vars)

    X = encoded_data.drop('ServiceSatisfaction', axis=1)
    y = encoded_data['ServiceSatisfaction']
    return X, y

model_df = df_cleaned.drop(['Date', 'FacilityCounty', 'FacilityOwnership'], axis=1)
X, y = feature_engineering(model_df)

# Model prediction
if st.button('Predict'):
    predictions = model.predict(X)
    X['Predictions'] = predictions  # Adding predictions to the dataframe

    # Display the dataframe with predictions
    # st.write(model_data)

    # Visualization of predictions
    count_fig = px.histogram(X, x='Predictions', title='Distribution of Predictions')
    st.plotly_chart(count_fig)

    # Feature importance can be visualized if relevant
    feature_importances = pd.DataFrame({'Feature': X.columns[:-1], 'Importance': model.feature_importances_})
    fi_fig = px.bar(feature_importances, x='Importance', y='Feature', orientation='h', title="Feature Importances")
    st.plotly_chart(fi_fig)