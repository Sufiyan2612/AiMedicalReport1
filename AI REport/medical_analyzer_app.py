import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import base64

# Constants
MAX_SAMPLE_SIZE = 1000  # For large dataset visualization
CACHE_TTL = 3600  # Cache time-to-live in seconds
DEFAULT_FILTERS = {
    'age_range': (0, 100),
    'conditions': [],
    'admission_types': [],
    'min_billing': 0,
    'max_billing': float('inf')
}

def get_download_link(df: pd.DataFrame, filename: str) -> str:
    """Generate a download link for a dataframe."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filters to the dataframe."""
    filtered_df = df.copy()
    
    # Age filter
    filtered_df = filtered_df[
        (filtered_df['Age'] >= filters['age_range'][0]) &
        (filtered_df['Age'] <= filters['age_range'][1])
    ]
    
    # Medical conditions filter
    if filters['conditions']:
        filtered_df = filtered_df[filtered_df['Medical Condition'].isin(filters['conditions'])]
    
    # Admission types filter
    if filters['admission_types']:
        filtered_df = filtered_df[filtered_df['Admission Type'].isin(filters['admission_types'])]
    
    # Billing amount filter
    filtered_df = filtered_df[
        (filtered_df['Billing Amount'] >= filters['min_billing']) &
        (filtered_df['Billing Amount'] <= filters['max_billing'])
    ]
    
    return filtered_df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the dataset by handling missing values and outliers."""
    # Create a copy to avoid modifying the original
    df_processed = df.copy()
    
    # Handle missing values
    numeric_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    
    # Fill numeric missing values with median
    for col in numeric_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].median())
    
    # Fill categorical missing values with mode
    for col in categorical_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
    
    # Handle outliers in numeric columns using IQR method
    for col in numeric_cols:
        Q1 = df_processed[col].quantile(0.25)
        Q3 = df_processed[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_processed[col] = df_processed[col].clip(lower_bound, upper_bound)
    
    return df_processed

def validate_input(age: int, gender: str, condition: str) -> bool:
    """Validate user input for patient analysis."""
    if not (1 <= age <= 100):
        st.error("Age must be between 1 and 100")
        return False
    if not gender or not isinstance(gender, str):
        st.error("Invalid gender selection")
        return False
    if not condition or not isinstance(condition, str):
        st.error("Invalid medical condition selection")
        return False
    return True

# Load the medical analyzer
@st.cache_data(ttl=CACHE_TTL)
def load_analyzer():
    try:
        if not os.path.exists('medical_analyzer.pkl'):
            st.error("Error: medical_analyzer.pkl file not found!")
            return None
        with open('medical_analyzer.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading analyzer: {str(e)}")
        return None

# Load data
@st.cache_data(ttl=CACHE_TTL)
def load_data():
    try:
        if not os.path.exists('healthcare_dataset.csv'):
            st.error("Error: healthcare_dataset.csv file not found!")
            return None
        df = pd.read_csv('healthcare_dataset.csv')
        required_columns = ['Age', 'Gender', 'Medical Condition', 'Test Results', 'Billing Amount', 'Admission Type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Error: Missing required columns: {', '.join(missing_columns)}")
            return None
        return preprocess_data(df)
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def get_sampled_data(df: pd.DataFrame, sample_size: int = MAX_SAMPLE_SIZE) -> pd.DataFrame:
    """Get a sampled version of the dataset for visualization."""
    if len(df) <= sample_size:
        return df
    return df.sample(n=sample_size, random_state=42)

def main():
    st.set_page_config(page_title="AI Medical Report Analyzer", layout="wide")
    
    # Initialize session state
    if 'filters' not in st.session_state:
        st.session_state.filters = DEFAULT_FILTERS.copy()
    
    st.title("🏥 AI Medical Report Analyzer Assistant")
    st.markdown("Intelligent system for analyzing medical reports and patient data")
    
    # Load data and analyzer
    df = load_data()
    analyzer = load_analyzer()
    
    if df is None or analyzer is None:
        st.error("Critical error: Unable to load required data or analyzer. Please check the error messages above.")
        return
    
    # Sidebar for navigation and settings
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose Analysis Type", 
                               ["Dashboard Overview", "Patient Analysis", "Medical Insights", "Report Generator"])
    
    # Add cache control in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")
    if st.sidebar.button("Clear Cache"):
        st.cache_data.clear()
        st.success("Cache cleared successfully!")
    
    # Add filters in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Filters")
    
    # Age range filter
    age_range = st.sidebar.slider(
        "Age Range",
        min_value=0,
        max_value=100,
        value=st.session_state.filters['age_range']
    )
    st.session_state.filters['age_range'] = age_range
    
    # Medical conditions filter
    conditions = st.sidebar.multiselect(
        "Medical Conditions",
        options=df['Medical Condition'].unique(),
        default=st.session_state.filters['conditions']
    )
    st.session_state.filters['conditions'] = conditions
    
    # Admission types filter
    admission_types = st.sidebar.multiselect(
        "Admission Types",
        options=df['Admission Type'].unique(),
        default=st.session_state.filters['admission_types']
    )
    st.session_state.filters['admission_types'] = admission_types
    
    # Billing amount filter
    min_billing, max_billing = st.sidebar.slider(
        "Billing Amount Range",
        min_value=float(df['Billing Amount'].min()),
        max_value=float(df['Billing Amount'].max()),
        value=(st.session_state.filters['min_billing'], st.session_state.filters['max_billing'])
    )
    st.session_state.filters['min_billing'] = min_billing
    st.session_state.filters['max_billing'] = max_billing
    
    # Apply filters
    filtered_df = apply_filters(df, st.session_state.filters)
    
    if page == "Dashboard Overview":
        st.header("📊 Medical Data Dashboard")
        
        # Export filtered data
        st.markdown(get_download_link(filtered_df, f"medical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), unsafe_allow_html=True)
        
        # Use sampled data for better performance
        sampled_df = get_sampled_data(filtered_df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", f"{len(filtered_df):,}")
        with col2:
            st.metric("Medical Conditions", len(filtered_df['Medical Condition'].unique()))
        with col3:
            st.metric("Average Age", f"{filtered_df['Age'].mean():.1f}")
        with col4:
            st.metric("Avg Billing", f"${filtered_df['Billing Amount'].mean():,.0f}")
        
        # Charts with sampled data
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(sampled_df['Medical Condition'].value_counts().reset_index(), 
                        x='index', y='Medical Condition',
                        title="Medical Conditions Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(sampled_df['Test Results'].value_counts().reset_index(), 
                        values='Test Results', names='index',
                        title="Test Results Distribution")
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Patient Analysis":
        st.header("👤 Individual Patient Analysis")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Patient Age", min_value=1, max_value=100, value=45)
        with col2:
            gender = st.selectbox("Gender", filtered_df['Gender'].unique())
        with col3:
            condition = st.selectbox("Medical Condition", filtered_df['Medical Condition'].unique())
        
        if st.button("Analyze Patient"):
            if not validate_input(age, gender, condition):
                return
                
            analysis = analyzer.analyze_patient_profile(age, gender, condition)
            
            if analysis:
                st.success("Analysis Complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Similar Cases", analysis['similar_cases'])
                with col2:
                    st.metric("Avg Billing", f"${analysis['avg_billing']:,.2f}")
                with col3:
                    st.metric("Common Medication", analysis['common_medication'])
                
                st.subheader("Test Result Patterns")
                test_df = pd.DataFrame(list(analysis['test_outcomes'].items()), 
                                     columns=['Result', 'Count'])
                fig = px.bar(test_df, x='Result', y='Count', 
                           title="Test Results for Similar Cases")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No similar cases found in database")
    
    elif page == "Medical Insights":
        st.header("🔍 Medical Data Insights")
        
        # Use sampled data for better performance
        sampled_df = get_sampled_data(filtered_df)
        
        # Age distribution by condition
        fig = px.box(sampled_df, x='Medical Condition', y='Age', 
                    title="Age Distribution by Medical Condition")
        st.plotly_chart(fig, use_container_width=True)
        
        # Billing analysis
        fig = px.box(sampled_df, x='Admission Type', y='Billing Amount',
                    title="Billing Amount by Admission Type")
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        numeric_cols = sampled_df.select_dtypes(include=['int64', 'float64']).columns
        if len(numeric_cols) > 1:
            corr_matrix = sampled_df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                          title="Correlation Matrix")
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "Report Generator":
        st.header("📋 Medical Report Generator")
        
        st.subheader("Generate Comprehensive Report")
        
        if st.button("Generate Summary Report"):
            with st.spinner("Generating report..."):
                summary = analyzer.generate_summary_report()
                
                st.markdown("### 📊 Database Summary")
                st.write(f"**Total Patients:** {summary['total_patients']:,}")
                
                st.markdown("### 🏥 Medical Conditions")
                conditions_df = pd.DataFrame(list(summary['conditions_distribution'].items()),
                                           columns=['Condition', 'Count'])
                st.dataframe(conditions_df, use_container_width=True)
                
                st.markdown("### 📈 Average Age by Condition")
                age_df = pd.DataFrame(list(summary['avg_age_by_condition'].items()),
                                    columns=['Condition', 'Average Age'])
                st.dataframe(age_df, use_container_width=True)
                
                st.markdown("### 🧪 Test Results Summary")
                test_df = pd.DataFrame(list(summary['test_results_summary'].items()),
                                     columns=['Result', 'Count'])
                st.dataframe(test_df, use_container_width=True)
                
                # Export report data
                report_data = pd.concat([
                    conditions_df,
                    age_df,
                    test_df
                ], axis=1)
                st.markdown(get_download_link(report_data, f"medical_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
