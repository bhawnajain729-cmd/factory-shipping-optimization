
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

st.title("Factory Reallocation & Shipping Optimization System")

df = pd.read_csv("dataset.csv")

df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df['Lead_Time'] = (df['Ship Date'] - df['Order Date']).dt.days

le_region = LabelEncoder()
le_ship = LabelEncoder()
df['Region_enc'] = le_region.fit_transform(df['Region'])
df['Ship_enc'] = le_ship.fit_transform(df['Ship Mode'])

X = df[['Region_enc','Ship_enc','Sales','Units']]
y = df['Lead_Time']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = RandomForestRegressor()
model.fit(X_train,y_train)

st.sidebar.header("User Controls")

product = st.sidebar.selectbox("Select Product",df["Product Name"].unique())
region = st.sidebar.selectbox("Select Region",df["Region"].unique())
ship = st.sidebar.selectbox("Ship Mode",df["Ship Mode"].unique())

sales = st.sidebar.slider("Sales Value",100,500,200)
units = st.sidebar.slider("Units",1,10,3)

region_enc = le_region.transform([region])[0]
ship_enc = le_ship.transform([ship])[0]

prediction = model.predict([[region_enc,ship_enc,sales,units]])

st.subheader("Predicted Lead Time")
st.write(f"Estimated Shipping Lead Time: **{prediction[0]:.2f} days**")

st.subheader("Factory Recommendation")

factory_scores = []
for f in df["Factory"].unique():
    simulated = prediction[0] - np.random.uniform(0,2)
    factory_scores.append((f, simulated))

factory_scores = sorted(factory_scores, key=lambda x: x[1])

rec_df = pd.DataFrame(factory_scores,columns=["Factory","Predicted Lead Time"])

st.dataframe(rec_df)

best = rec_df.iloc[0]

st.success(f"Recommended Factory: {best['Factory']}")
