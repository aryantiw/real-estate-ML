# import streamlit as st
# import pickle
# import json
# import numpy as np

# # Load the trained model
# with open('banglore_home_prices_model.pickle', 'rb') as f:
#     model = pickle.load(f)

# # Load the columns information
# with open('columns.json', 'r') as f:
#     columns = json.load(f)
#     data_columns = columns['data_columns']

# # Function to predict house price
# def predict_price(location, total_sqft, bath, bhk):
#     loc_index = np.where(np.array(data_columns) == location)[0][0]

#     x = np.zeros(len(data_columns))
#     x[0] = total_sqft
#     x[1] = bath
#     x[2] = bhk
#     if loc_index >= 0:
#         x[loc_index] = 1

#     return model.predict([x])[0]

# # Streamlit App
# def main():
#     st.title('Bangalore House Price Prediction')

#     # User inputs
#     location = st.selectbox('Location', data_columns)
#     total_sqft = st.number_input('Total Square Feet Area')
#     bath = st.number_input('Number of Bathrooms')
#     bhk = st.number_input('Number of Bedrooms')

#     # Predict button
#     if st.button('Predict Price'):
#         result = predict_price(location, total_sqft, bath, bhk)
#         st.success(f'Predicted Price: {result:.2f} Lakhs')

# if __name__ == '__main__':
#     main()


import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd

# Load the trained model
with open('banglore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

# Load the columns information
with open('columns.json', 'r') as f:
    columns = json.load(f)
    data_columns = columns['data_columns']

# Function to predict house price
def predict_price(location, total_sqft, bath, bhk):
    loc_index = np.where(np.array(data_columns) == location)[0][0]

    x = np.zeros(len(data_columns))
    x[0] = total_sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]

# Streamlit App
def main():
    st.set_page_config(page_title='Bangalore House Price Prediction', page_icon='🏠', layout='wide')

    # Title and description
    st.title('Bangalore House Price Prediction')
    st.write('This app predicts the price of a house in Bangalore based on location, total square feet area, number of bathrooms, and number of bedrooms.')

    # Image
    st.image('https://images.unsplash.com/photo-1551288049-bebda4e38f71', use_column_width=True)

    # User inputs
    st.sidebar.header('Enter House Details')
    location = st.sidebar.selectbox('Location', data_columns)
    total_sqft = st.sidebar.number_input('Total Square Feet Area', min_value=100, max_value=10000, step=100, value=1000)
    bath = st.sidebar.number_input('Number of Bathrooms', min_value=1, max_value=10, step=1, value=2)
    bhk = st.sidebar.number_input('Number of Bedrooms', min_value=1, max_value=10, step=1, value=2)

    # Predict button
    if st.sidebar.button('Predict Price'):
        result = predict_price(location, total_sqft, bath, bhk)
        st.success(f'Predicted Price: ₹ {result:.2f} Lakhs')

if __name__ == '__main__':
    main()
