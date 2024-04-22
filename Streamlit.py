import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained machine learning model
with open('dt.pkl', 'rb') as file:
    model = pickle.load(file)
with open('lr.pkl', 'rb') as file:
    model = pickle.load(file)
with open ('rf.pkl','rb') as file:
    model = pickle.load(file)
    

# Define the Streamlit app
def main():
    # Set the title and description
    st.title('Resale Price Predictor')
    st.write('Enter details of the flat to predict its resale price.')

    # User input fields
    town = st.selectbox('Town', range(27))  # Assuming 27 towns based on your provided data
    flat_type = st.selectbox('Flat Type', range(6))  # Assuming 6 flat types based on your provided data
    block = st.number_input('Block', min_value=0)
    street_name = st.number_input('Street Name', min_value=0)
    storey_range = st.number_input('Storey Range', min_value=0)
    flat_model = st.number_input('Flat Model', min_value=0)
    lease_commence_date = st.number_input('Lease Commence Date', min_value=1960, max_value=2023)
    reg_year = st.number_input('Registration Year', min_value=1990, max_value=2024)
    reg_month = st.number_input('Registration Month', min_value=1, max_value=12)

    # Button to trigger prediction
    if st.button('Predict Resale Price'):
        # Store user inputs in a dictionary
        user_input = {'town': town,
                      'flat_type': flat_type,
                      'block': block,
                      'street_name': street_name,
                      'storey_range': storey_range,
                      'flat_model': flat_model,
                      'lease_commence_date': lease_commence_date,
                      'reg_year': reg_year,
                      'reg_month': reg_month}
        
        # Convert user input dictionary to DataFrame
        input_df = pd.DataFrame([user_input])
        
        # Predict resale price
        predicted_price = model.predict(input_df)
        
        # Display predicted price
        st.write(f'Predicted Resale Price: ${predicted_price[0]:.2f}')

if __name__ == '__main__':
    main()
