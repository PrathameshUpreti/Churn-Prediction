#using stearmlit
import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder

#load the model
model=tf.keras.models.load_model('model.h5')

#load the encoder and scalar
with open('label_encoder_gender.pkl','rb') as file:
    gender_encoder=pickle.load(file)
    
with open('onehot_geo.pkl','rb') as file:
    geo_encoder=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

    ##streamlit app

st.title("Coutomer Churn Predictoion Model")

#user input in stramlit

geography=st.selectbox('Geography',geo_encoder.categories_[0])
gender=st.selectbox('Gender',gender_encoder.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estomated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Num of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

 #Prepare input data
input_data = pd.DataFrame( {
    'CreditScore': [credit_score],
    'Geography':[geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

#one hot code encoder
geography_encoder=geo_encoder.transform(input_data[['Geography']]).toarray()
geography_encoder_df=pd.DataFrame(geography_encoder, columns=geo_encoder.get_feature_names_out(['Geography']))

#combine columns with input
input_data = input_data.drop('Geography', axis=1)
input_data=pd.concat([input_data.reset_index(drop=True),geography_encoder_df],axis=1)

# Encode Gender
input_data['Gender'] = gender_encoder.transform(input_data['Gender'])
#Scale the input data

input_data_scale=scaler.transform(input_data)
#predictiom
prediction=model.predict(input_data_scale)
prediction_prob=prediction[0][0]

if prediction_prob>0.5:
    st.write('The coustomer is likely to churn')
else:
    st.write('Not likely to churn')

