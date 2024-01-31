import streamlit as st
import pickle
import pandas as pd
import requests
from io import BytesIO

# Function to load model from URL
def load_model_from_url(url):
    response = requests.get(url)
    model = pickle.load(BytesIO(response.content))
    return model

# Load the trained model
model_url = 'https://github.com/harxhpatel19/heart_disease/raw/main/banglore_home_prices_model.pickle'
model = load_model_from_url(model_url)

# Function to load dataset from URL
def load_dataset_from_url(url):
    response = requests.get(url)
    df = pd.read_csv(BytesIO(response.content))
    return df

# Load the dataset used for training
df_url = 'https://github.com/harxhpatel19/heart_disease/raw/main/bhp.csv'
df = load_dataset_from_url(df_url)

# Create a Streamlit web app
st.title('Bangalore House Price Prediction')

# Sidebar with input fields
st.sidebar.header('Enter Property Details')

# Location
locations = df.columns[3:]
location = st.sidebar.selectbox('Location', locations)

# Total Square Feet
sqft = st.sidebar.slider('Total Square Feet', min_value=100, max_value=10000, value=1000, step=100)

# Number of Bedrooms (BHK)
bhk = st.sidebar.number_input('Number of Bedrooms (BHK)', min_value=1, max_value=10, value=2)

# Number of Bathrooms
bath = st.sidebar.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)

# Predict the price
def predict_price(location, sqft, bhk, bath):
    loc_index = df.columns.get_loc(location)
    x = [sqft, bath, bhk] + [0] * (len(df.columns) - 3)
    if loc_index >= 0:
        x[loc_index] = 1
    price = model.predict([x])[0]
    return price

if st.sidebar.button('Predict Price'):
    predicted_price = predict_price(location, sqft, bhk, bath)
    st.sidebar.subheader(f'Predicted Price: ₹{predicted_price:.2f} Lakhs')

# Data Exploration
st.subheader('Data Exploration')
st.write(df.head(10))

# Display scatter plot
st.subheader('Scatter Plot for Price vs. Square Feet')
st.scatter_chart(df[['total_sqft', 'price']])

# Display histogram for price per square feet
st.subheader('Histogram for Price Per Square Feet')
st.bar_chart(df['price_per_sqft'])

# Display histogram for number of bedrooms
st.subheader('Histogram for Number of Bedrooms')
st.bar_chart(df['bhk'])

# Display histogram for number of bathrooms
st.subheader('Histogram for Number of Bathrooms')
st.bar_chart(df['bath'])

# Display a link to the dataset
st.subheader('Download the Dataset')
st.markdown('[Download the dataset](https://www.kaggle.com/amitabhajoy/bengaluru-house-price-data)')

# Display information about the project
st.subheader('About')
st.write('This web app provides a simple interface for predicting house prices in Bangalore based on location, square feet area, number of bedrooms, and number of bathrooms.')

# Run the Streamlit app
if __name__ == '__main__':
    st.set_page_config(page_title="Bangalore House Price Prediction", page_icon="🏠", layout="wide")
    st.run()
