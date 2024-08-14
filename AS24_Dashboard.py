# import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px

#set page confic
st.set_page_config(page_title="AutoScout24 Dashboard", page_icon="🚗", layout="wide", initial_sidebar_state="expanded")

# Load the data
data = pd.read_csv('autoscout24.csv', sep=';')
data = data.dropna()
data.reset_index(drop=True, inplace=True)

# Variable for the sidebar selection
gb_make_to_model = data.groupby('make')["model"].unique()
gb_model_to_fuel = data.groupby('model')["fuel"].unique()
gb_model_to_year = data.groupby('model')["year of sale"].unique()
max_hp_of_model = data.groupby('model')["hp"].max()
min_hp_of_model = data.groupby('model')["hp"].min()

# Title
st.title('AutoScout24 Dashboard')

##### Sidebar ####

# must have selection:
st.sidebar.title('Navigation and Model Selection')
make = st.sidebar.selectbox('Select the brand', data['make'].unique())
model = st.sidebar.selectbox('Select the car model', np.sort(gb_make_to_model[make]))

# optional selection:
fuel_on = st.sidebar.toggle('Optional selection for Fuel', False)
if fuel_on:
    fuel = st.sidebar.selectbox('Select the fuel type', options=(gb_model_to_fuel[model]))
else:
    fuel = 'All'
year_on = st.sidebar.toggle('Optional selection for Year', False)
if year_on:
    year = st.sidebar.selectbox('Select the year of sale', options=np.sort(gb_model_to_year[model]))
else:
    year = 'All'   

min_hp = st.sidebar.slider('Select the minimum horsepower you want', min_value=int(min_hp_of_model[model]), max_value=int(max_hp_of_model[model]), value=0, step=10)

# Create a new dataframe based on the selection
if fuel == 'All' and year == 'All':
    selected_data = data[(data['make'] == make) & (data['model'] == model) & (data['hp'] >= min_hp)]
elif fuel == 'All' and year != 'All':
    selected_data = data[(data['make'] == make) & (data['model'] == model) & (data['year of sale'] == year) & (data['hp'] >= min_hp)]
elif fuel != 'All' and year == 'All':
    selected_data = data[(data['make'] == make) & (data['model'] == model) & (data['fuel'] == fuel) & (data['hp'] >= min_hp)]
else:
    selected_data = data[(data['make'] == make) & (data['model'] == model) & (data['fuel'] == fuel) & (data['year of sale'] == year) & (data['hp'] >= min_hp)]

selected_data.reset_index(drop=True, inplace=True)
selected_data.index.name = 'Index'

#####   Main page #####
tab1, tab2, tab3 = st.tabs(["Statistics", "Machine Learning","Sunnburst Overview"])

# Statistics
with tab1:
    st.subheader('Sales statistics of the selected cars:')
    st.write("We have found", selected_data.shape[0], "cars that match your selection.")
    #Boxplot
    st.plotly_chart(px.box(selected_data, x='price', y='model', title='Boxplot of the selected cars'))
    #Scatterplot
    st.plotly_chart(px.scatter(selected_data['price'], 
                               title="Solded cars on AutoScout24", 
                               trendline='ols'))
    #Dataframe
    st.write("Dataframe (based on your selection)")
    st.write(selected_data)

# Machine Learning
with tab2:
    st.header('Machine Learning')
    st.write("In this section, I provide a brief overview of the Exploratory Data Analysis (EDA) and the key metrics involved in a regression-based Machine Learning project. Please note that this section does not include a prediction function using the trained model, as the pricing depends on many more variables (features) than those included in the model. To avoid creating misleading impressions or expectations, I have chosen not to offer this feature here.")
    heatmap = sns.heatmap(data[["mileage","hp","year of sale","price"]].corr(), annot=True)
    st.subheader("Correlation Matrix")
    st.pyplot(heatmap.get_figure())

    # dict with number of cars sold per make
    make_dict = { i: data[data["make"]==i].shape[0] for i in data["make"].unique()}
    # get the top 5 brands by number of cars sold
    top_5_brands = sorted(make_dict, key=make_dict.get, reverse=True)[:5]
    top_5 = data[data["make"].isin(top_5_brands)]
    top_5.reset_index(drop=True, inplace=True)

    st.subheader("Like in the Task mentioned, I will focus on the top 5 brands by number of cars sold.")
    st.write("The top 5 brands are:", top_5_brands)
    st.plotly_chart(px.histogram(top_5, x='price', title='Histogram of the price of the top 5 brands'))

    st.subheader("Machine Learning Metrics")
    st.write("The trained model is a Random Forest Regressor and performed with the following metrics:")
    st.write("Mean Squared Error: 5105570.0142824715")
    st.write("Mean Absolute Error: 1313.0174672063786")
    st.write("R2 Score: 0.9408354526238328")
    st.image("predictive_true_comp.png", caption="Comparison of the predicted and true prices")


# Sunburst chart
with tab3:
    st.write("This Chart let you explore the sales volume per brand in a sunburst chart.")
    st.write("(only use the 'brand' selection from the sidebar)")
    fig = px.sunburst(
        data_frame= data[data['make'] == make],
        path=['make', 'model','offerType', 'fuel', 'year of sale'],
        values='price', 
        #title='Sunburst chart of the selected cars',
        height=850,
        width=850
        )
    st.plotly_chart(fig)