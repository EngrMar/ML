from flask import Flask, render_template
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from flask import request
sns.set()
import math

raw_data = pd.read_csv(r"C:\Users\marion.deguzman\PycharmProjects\ML_LR\CAR_DETAILS_FROM_CAR_DEKHO.csv")
app = Flask(__name__)


@app.route('/')
def home_page():
    q = raw_data['selling_price'].quantile(0.99)
    data_1 = raw_data[raw_data['selling_price'] < q]
    q2 = data_1['km_driven'].quantile(0.99)
    data_2 = data_1[data_1['km_driven'] < q2]
    data_3 = data_2.drop(['name'], axis=1)
    data_3 = data_3.drop(['fuel'], axis=1)
    data_3 = data_3.drop(['seller_type'], axis=1)
    data_3 = data_3.drop(['transmission'], axis=1)
    data_3 = data_3.drop(['owner'], axis=1)
    data_cleaned = data_3.reset_index(drop=True)
    log_price = np.log(data_cleaned['selling_price'])
    data_cleaned['log_price'] = log_price
    data_cleaned = data_cleaned.drop(['selling_price'], axis=1)
    targets = data_cleaned['log_price']
    inputs = data_cleaned.drop(['log_price'], axis=1)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(inputs)
    inputs_scaled = scaler.transform(inputs)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=42)

    reg = LinearRegression()
    reg.fit(x_train, y_train)
    y_hat = reg.predict(x_train)
    y_hat_test = reg.predict(x_test)
    y_test = y_test.reset_index(drop=True)
    df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Predictions'])
    df_pf['Target'] = np.exp(y_test)
    df_pf['Residual'] = df_pf['Target'] - df_pf['Predictions']
    df_pf['Difference%'] = np.abs(df_pf['Residual'] / df_pf['Target'] * 100)

    ####
    #Year_of_Purchase = input("What year was the car bought?\n")
    #Mileage = input('What is the mileage in km?\n')
    #new_input = [[float(Year_of_Purchase), float(Mileage)]]
    #new_input_scaled = scaler.transform(new_input)
    #y_hat_new = reg.predict(new_input_scaled)
    #y_hat_new_in = np.exp(y_hat_new)
    return render_template("index.html")

@app.route("/Prediction")
def Prediction():
    yr_of_pr = request.args['year_of_pur']
    mileage = request.args['mil']
    q = raw_data['selling_price'].quantile(0.99)
    data_1 = raw_data[raw_data['selling_price'] < q]
    q2 = data_1['km_driven'].quantile(0.99)
    data_2 = data_1[data_1['km_driven'] < q2]
    data_3 = data_2.drop(['name'], axis=1)
    data_3 = data_3.drop(['fuel'], axis=1)
    data_3 = data_3.drop(['seller_type'], axis=1)
    data_3 = data_3.drop(['transmission'], axis=1)
    data_3 = data_3.drop(['owner'], axis=1)
    data_cleaned = data_3.reset_index(drop=True)
    log_price = np.log(data_cleaned['selling_price'])
    data_cleaned['log_price'] = log_price
    data_cleaned = data_cleaned.drop(['selling_price'], axis=1)
    targets = data_cleaned['log_price']
    inputs = data_cleaned.drop(['log_price'], axis=1)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(inputs)
    inputs_scaled = scaler.transform(inputs)

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2, random_state=42)

    reg = LinearRegression()
    reg.fit(x_train, y_train)
    y_hat = reg.predict(x_train)
    y_hat_test = reg.predict(x_test)
    y_test = y_test.reset_index(drop=True)
    df_pf = pd.DataFrame(np.exp(y_hat_test), columns=['Predictions'])
    df_pf['Target'] = np.exp(y_test)
    df_pf['Residual'] = df_pf['Target'] - df_pf['Predictions']
    df_pf['Difference%'] = np.abs(df_pf['Residual'] / df_pf['Target'] * 100)

    ####
    #Year_of_Purchase = input("What year was the car bought?\n")
    #Mileage = input('What is the mileage in km?\n')
    new_input = [[float(yr_of_pr), float(mileage)]]
    new_input_scaled = scaler.transform(new_input)
    y_hat_new = reg.predict(new_input_scaled)
    y_hat_new_in = round(float(np.exp(y_hat_new)), 2)
    return render_template("index_2.html", output=y_hat_new_in)
if __name__ == "__main__":
    app.run(debug=True)