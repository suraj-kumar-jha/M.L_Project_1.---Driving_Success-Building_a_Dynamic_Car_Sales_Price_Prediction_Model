## Driving Success: Building a Dynamic Car Sales Price Prediction Model 
---------------------------------------------------------------------------

------------------------------------------------------------------
### TOOLS USED IN THE PROJECT:-

------------------------------------------------------------------

#### 1. Jupyter Notebook:-
Jupyter Notebook served as the interactive development environment where we wrote and executed our code. 
It allowed us to document our progress, visualize data, and iterate on our models efficiently.

#### 2. Python:-
Python was the primary programming language used for this project. Its extensive libraries and frameworks for data analysis and machine learning made it an ideal choice for developing our predictive models.

#### 3. Machine Learning Algorithms:-
We employed various machine learning algorithms to predict car sales prices, including:

=> Linear Regression (LR),
=> Decision Tree Regressor (DTR),
=> Random Forest Regressor (RFR).

#### 4. Data:-
The dataset used in this project comprised historical data on used car sales, including features such as make, model, year, mileage, and price. 
This data was crucial for training and validating our machine learning models.

------------------------------------------------------------------

### TABLE OF CONTENTS:-

------------------------------------------------------------------

- [IMPORT LIBRARIES](#import-libraries)
- [IMPORT DATASET](#import-dataset)
- [DATA INFORMATIONS](#data-informations)
- [DATA PREPROCESSING](#data-preprocessing)
- [DATA ANALYSIS](#data-analysis)
- [EXPLORATORY DATA ANALYSIS](#exploratory-data-analysis)
- [1. LINEAR REGRESSION MODEL](#1.linear-regression-model)
- [2. DECISION TREE REGRESSOR MODEL](#2.decision-tree-model)
- [3. RANDOM FOREST REGRESSOR MODEL](#4.random-forest-regressor-model)
- [CONCLUSION](#conclusion)

------------------------------------------------------------------

### Problem Statement:-
------------------------------------------------------------------

Develop a machine learning model to predict the selling price of used cars based on a dataset containing 8128 entries and 13 features. 
With details such as car name, year,selling price, kilometers driven, fuel type, seller type, transmission, owner history, mileage, engine capacity, maximum power, and seating capacity, the goal is to create a reliable model for estimating the fair market value of
used cars. Key challenges include handling diverse features, addressing missing values, selecting appropriate algorithms, and optimizing model performance for accurate predictions, ensuring applicability in real-world scenarios for buyers and sellers.

------------------------------------------------------------------

### IMPORT LIBRARIES
------------------------------------------------------------------

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
```

------------------------------------------------------------------

### IMPORT DATASET
------------------------------------------------------------------

```
car_data = pd.read_csv("/Users/pushkarnathsingh/Downloads/ML Projects (Y)/Project DATA SETS/Project - 1/car sales data/dataset/Cardetails.csv")
car_data.head(8)
```
<img width="979" alt="Screenshot 2024-05-22 at 2 46 11 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/d7ebc062-da96-4040-90ef-857b96461b3a">

------------------------------------------------------------------

### DATA INFORMATIONS

------------------------------------------------------------------

```
print(f"Data_Shape = {car_data.shape}")
print("\n################################\n")
print(f"Data_Describe\n\n = {car_data.describe()}")
print("\n################################\n")
print(f"Data_Info\n\n = {car_data.info()}")
```
<img width="549" alt="Screenshot 2024-05-22 at 2 48 58 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/147ce7b3-0216-4d14-bce7-a03a5c322e62">

------------------------------------------------------------------

### DATA PREPROCESSING

------------------------------------------------------------------

#### 1. Check and Remove Duplicate Values =>
```
print(f" Duplicate_Values = {car_data.duplicated().sum()}")
```
=>  Duplicate_Values = 1202

#### Drop the duplicate values =>
```
car_data.drop_duplicates(inplace = True)
```

#### Check and Remove Duplicate Values =>
```
print(f" Duplicate_Values = {car_data.duplicated().sum()}")
```
=> Duplicate_Values = 0

```

```

#### 2. Check and Remove Null Values =>
```
print(f"Total_NullValues => \n{car_data.isnull().sum()}")
```
<img width="201" alt="Screenshot 2024-05-22 at 2 56 28 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/7d07162e-0e4b-4a2c-a536-7f593ed84b3e">

#### Drop the Null Values =>
```
car_data.dropna(inplace = True)
```
#### Check and Remove Null Values =>
```
print(f"Total_NullValues => \n{car_data.isnull().sum()}")
```
<img width="203" alt="Screenshot 2024-05-22 at 3 03 50 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/29032d7c-22d2-4a34-b133-93419fa4486a">


```
print("Data_Shape = ", car_data.shape)
```
=> Data_Shape =  (6717, 13)

```

```

#### 3. Remove Unwanted columns:-
```
car_data.drop(columns='torque',inplace = True)
```

<img width="956" alt="Screenshot 2024-05-22 at 3 06 46 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/ce04e765-bf9b-489d-8540-5e72917a1639">

```

```

#### 4. Remove extra str Values from columns:-

```
def Remove_extra_str(col_name):
    col_name = col_name.split()[0]
    return col_name.strip(" ")
```

```
car_data['name'] = car_data['name'].apply(Remove_extra_str)
car_data['mileage'] = car_data['mileage'].apply(Remove_extra_str)
car_data['engine'] = car_data['engine'].apply(Remove_extra_str)
car_data['max_power'] = car_data['max_power'].apply(Remove_extra_str)
```
```

```
#### 5. CHANGE DATA TYPES OF COLUMNS:-

```
car_data['seats'] = car_data['seats'].astype('int')
car_data['mileage'] = car_data['mileage'].astype('float').round()
car_data['mileage'] = car_data['mileage'].astype('int')
car_data['engine'] = car_data['engine'].astype('int')
car_data['max_power'] = car_data['mileage'].astype('int')
```
```
car_data.head()
```
<img width="812" alt="Screenshot 2024-05-22 at 3 12 47 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/3210ec5e-4741-4355-b88e-a1e087b32627">

------------------------------------------------------------------

### DATA ANALYSIS

------------------------------------------------------------------

```
for col in car_data.columns:
    print(f"Unique_Values => \n{col}")
    print(f"{car_data[col].unique()}")
    print("\n#######################\n")
```

<img width="364" alt="Screenshot 2024-05-22 at 3 15 35 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/dcb7bcf5-dbb5-4181-9382-69eda695da7b">


<img width="362" alt="Screenshot 2024-05-22 at 3 21 38 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/bd616b60-2eee-4100-b981-b2fd78bb0925">

------------------------------------------------------------------

### EXPLORATORY DATA ANALYSIS

------------------------------------------------------------------

#### 1. What are the top 5 car types present in the dataset?

```
top_cartypes = car_data['name'].value_counts().nlargest(5)
top_cartypes
```
<img width="232" alt="Screenshot 2024-05-22 at 3 24 36 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/f32c797e-016a-4835-b072-414a78fccc4d">

```
# Bar Plot:-
plot = top_cartypes.plot(kind = 'bar')

# Title
plt.title('Top 5 Car_Types in dataset:-')

# Adding values to each bar:
for i,v in enumerate(top_cartypes):
    plot.text(i,v + 0.1, str(v), ha = 'center',va = 'bottom')
plt.show()
```
<img width="583" alt="Screenshot 2024-05-22 at 3 26 00 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/d43f81a9-7fd5-452b-8627-3f235d229c4e">

------------------------------------------------------------------

#### 2. What are the trends over time regarding the selling prices of cars in the dataset?

```
# Group by 'year' and calculate the mean selling price for each year

mean_price_by_year = car_data.groupby('year')['selling_price'].mean()
mean_price_by_year
```
<img width="312" alt="Screenshot 2024-05-22 at 3 28 18 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/7dc87fab-087c-4467-9e0b-dcff54174717">

```
# Plotting the trend over time

plt.figure(figsize=(10, 6))
mean_price_by_year.plot()
plt.title('Mean Selling Price Trend Over Time')
plt.xlabel('Year')
plt.ylabel('Mean Selling Price')
plt.grid(True)
plt.show()
```
<img width="738" alt="Screenshot 2024-05-22 at 3 30 19 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/0d543c97-8049-4a93-a653-f67a381cdf5b">

#### => The analysis indicates a consistent upward trend in mean selling prices of cars over time, with fluctuations and accelerated growth observed in recent years. This suggests changing market dynamics and increasing consumer demand for higher-priced vehicles.

------------------------------------------------------------------

#### 3. What is the average selling price difference between different fuel types in the dataset, and how does it influence the pricing of cars?

```
# Grouping data by fuel type and calculating the mean selling price

fuel_type_analysis = car_data.groupby('fuel')['selling_price'].mean().reset_index()
fuel_type_analysis = fuel_type_analysis.sort_values(by='selling_price',ascending=False)
fuel_type_analysis
```
<img width="178" alt="Screenshot 2024-05-22 at 3 35 03 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/4d1d86e1-c76f-4cc2-a01c-450f3e91edf6">

```
plot = fuel_type_analysis.plot(kind = 'bar')
plot
```
<img width="353" alt="Screenshot 2024-05-22 at 3 37 16 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/0c4c2341-7481-41ef-95f6-bd19105bd512">

#### => Diesel cars command the highest average selling prices, likely due to their perceived fuel efficiency and performance. Petrol cars, while also popular, have notably lower average selling prices compared to diesel. CNG and LPG cars, being alternative fuel options, generally have lower selling prices, possibly due to factors such as availability of refueling stations and market demand.

------------------------------------------------------------------

#### 4. What is the average selling price for cars with automatic transmission compared to those with manual transmission?

```
# Grouping data by transmission and calculating the mean selling price

transmission = car_data.groupby('transmission')['selling_price'].mean().reset_index()
transmission_plot = transmission.plot(kind = 'bar')
print(transmission),transmission_plot
```

<img width="278" alt="Screenshot 2024-05-22 at 4 11 31 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/b42da773-ddf0-45c1-bc89-5fe2b850a8b8">


<img width="568" alt="Screenshot 2024-05-22 at 4 02 20 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/4da488ed-72ae-402a-91f1-7479e89d377a">

* The analysis reveals a notable discrepancy in selling prices between cars with automatic and manual transmissions.
  
* Automatic transmission cars command a higher average selling price compared to manual transmission cars.
  
* This observation underscores the influential role of transmission type in determining the market value of cars, with automatic transmissions generally demanding a premium over manual options.

------------------------------------------------------------------

#### 5. What is the average selling price trend over time for different seller types?
```
# Grouping data by fuel type and calculating the mean selling price:-

seller_type = car_data.groupby('seller_type')['selling_price'].mean().reset_index()

# Plot:-
seller_type_plot = seller_type.plot(kind = 'bar')
print(seller_type),seller_type_plot
```

<img width="330" alt="Screenshot 2024-05-22 at 4 10 40 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/2105f8d3-7465-4b81-94f2-d5b7d38d1f63">


<img width="604" alt="Screenshot 2024-05-22 at 4 06 44 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/1f15b04c-262f-4c1d-b879-a44d9057a981">

* Dealerships tend to have significantly higher average selling prices compared to individual sellers and trustmark dealers.

* Individual sellers offer cars at a lower average selling price compared to dealerships, indicating potential cost savings for buyers when purchasing     from individual sellers.

* Trustmark dealerships fall between individual sellers and dealerships in terms of average selling price, suggesting a middle ground for buyers seeking   a balance between price and dealer credibility.

* The choice of seller type can significantly impact the selling price of a car, with dealerships commanding a premium for their services and
  reputation, while individual sellers offer more competitive prices.

------------------------------------------------------------------

#### ONE - HOT ENCODING -> TO CONVERT CATEGORICAL DATA TO NUMERICAL FORMAT:-

------------------------------------------------------------------

```
car_data['name'].unique()
```
```
=>  array(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'], dtype=object)
```
```
car_data['name'] = car_data['name'].replace(['Maruti', 'Skoda', 'Honda', 'Hyundai', 'Toyota', 'Ford', 'Renault',
       'Mahindra', 'Tata', 'Chevrolet', 'Datsun', 'Jeep', 'Mercedes-Benz',
       'Mitsubishi', 'Audi', 'Volkswagen', 'BMW', 'Nissan', 'Lexus',
       'Jaguar', 'Land', 'MG', 'Volvo', 'Daewoo', 'Kia', 'Fiat', 'Force',
       'Ambassador', 'Ashok', 'Isuzu', 'Opel'],
                                           [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])
```
```

```

```
car_data['fuel'].unique()
```
```
=>  array(['Diesel', 'Petrol', 'LPG', 'CNG'], dtype=object)
```
```
car_data['fuel'] = car_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'],[1,2,3,4])
```
```

```
```
car_data['seller_type'].unique()
```
```
=>  array(['Individual', 'Dealer', 'Trustmark Dealer'], dtype=object)
```
```
car_data['seller_type'] = car_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'],[1,2,3])
```
```

```
```
car_data['transmission'].unique()
```
```
=>  array(['Manual', 'Automatic'], dtype=object)
```
```
car_data['transmission'] = car_data['transmission'].replace(['Manual', 'Automatic'],[1,2])
```
```

```
```
car_data['owner'].unique()
```
```
=>  array(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'], dtype=object)
```
```
car_data['owner'] = car_data['owner'].replace(['First Owner', 'Second Owner', 'Third Owner',
       'Fourth & Above Owner', 'Test Drive Car'],[1,2,3,4,5])
```
```

```
```
car_data.info()
```
<img width="380" alt="Screenshot 2024-05-22 at 4 26 18 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/7910b1f4-dcca-4b32-bf9d-2a9e06acd06b">

---------------------------------------------------

#### RESET INDEX and Remove present old index:-

```
car_data.reset_index(inplace = True)
```
```
car_data.drop(columns='index',inplace = True)
car_data.head(5)
```

<img width="754" alt="Screenshot 2024-05-22 at 4 29 07 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/9dc3a530-93a0-438c-869f-39a22699f004">

----------------------------------------------------------------------

#### Now Dividing Columns into Independent(input) and Dependent(output) variable:

```
x = car_data.drop(columns='selling_price')
y = car_data['selling_price']
```
-----------------------------------------------------------------------

#### 1. LINEAR REGRESSION MODEL

```
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=18)

LR_Model = LinearRegression()

LR_Model.fit(x_train,y_train)
```
```
LR_Prediction = LR_Model.predict(x_test)
LR_Prediction

=>  array([594839.15303156, 298604.63237901, 476828.24727747, ...,
       666416.06150757, 351455.359338  ,  45099.63197426])
```
#### => Accuracy Score of the LinearRegression_Model:-

```
Model_Score = r2_score(y_test,LR_Prediction)

print(f"LINEAR_REGRESSION_MODEL_SCORE = {Model_Score}")
```

=>  LINEAR_REGRESSION_MODEL_SCORE = 0.5902072494846233

------------------------------------------------------------------------

### 2. DECISION TREE REGRESSOR MODEL
```
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=20)

DTR_Model = DecisionTreeRegressor()

DTR_Model.fit(X_train,Y_train)
```
```
DTR_Prediction = DTR_Model.predict(X_test)
DTR_Prediction

=>  array([1650000.,  675000.,  229999., ...,  260000.,  400000., 2100000.])
```

#### => Accuracy Score of the DTR_Model:-
```
DTR_Score = r2_score(Y_test,DTR_Prediction)

print(f"DECISION_TREE_REGRESSOR_MODEL_SCORE = {DTR_Score}")
```

=>  DECISION_TREE_REGRESSOR_MODEL_SCORE = 0.888219688046386 

--------------------------------------------------------------------------

### 3. RANDOM FOREST REGRESSOR MODEL

```
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.2,random_state=96)

RFR_Model = RandomForestRegressor()

RFR_Model.fit(X_train,Y_train)
```
```
RFR_Prediction = RFR_Model.predict(X_test)
RFR_Prediction

=>  array([ 118397.59      ,  351783.30333333,  518679.97      , ...,
        102913.31666667,  354855.        , 1943880.        ])
```

#### => Accuracy Score of the RFR_Model:-
```
RFR_Score = r2_score(Y_test,RFR_Prediction)

print(f"RANDOM_FOREST_REGRESSOR_MODEL_SCORE = {RFR_Score}")
```
=>  RANDOM_FOREST_REGRESSOR_MODEL_SCORE = 0.904361560997204

```

```
### ACCURACY SCORES OF THE MACHINE LEARNING MODELS
```
print(f"\n=> ACCURACY SCORES OF THE MACHINE LEARNING MODELS:-\n")
print(f"1. LINEAR_REGRESSION_MODEL_SCORE = {Model_Score}")
print("\n----------------------------------------------------------------\n")
print(f"2. DECISION_TREE_REGRESSOR_MODEL_SCORE = {DTR_Score}")
print("\n----------------------------------------------------------------\n")
print(f"3. RANDOM_FOREST_REGRESSOR_MODEL_SCORE = {RFR_Score}")
```
<img width="584" alt="Screenshot 2024-05-22 at 5 26 32 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/41376c15-2f57-472c-9160-b302c68b5767">


```

```
=> Based on the comprehensive analysis conducted, it is evident that the Random Forest Regressor (RFR) model has significantly outperformed the Decision Tree Regressor (DTR) model, achieving a remarkable score of 90%. In comparison, the DTR model scored 85%, while the Linear Regression (LR) model lagged behind with a score of only 59%. This substantial difference in performance highlights the superior predictive accuracy of the RFR model. Consequently, we have decided to adopt the RFR model for predicting car sales prices, as it promises to deliver more reliable and precise estimations.

-----------------------------------------------------------------------------------

#### Now checking Model prediction with a demo dataset:
```
demo_dataset = pd.DataFrame([[1,2011,30000,2,1,1,1,21,998,21,5]],columns=["name","year","km_driven","fuel","seller_type","transmission","owner","mileage","engine","max_power","seats"])
demo_dataset
```
<img width="661" alt="Screenshot 2024-05-22 at 4 46 13 AM" src="https://github.com/suraj-kumar-jha/Machine-Deep-Learning-Projects/assets/155900363/50ed2c42-bfb8-492d-af1f-36162e243bc8">

```
new_prediction = RFR_Model.predict(demo_dataset)
new_prediction
```
=> array([241891.32333333])

----------------------------------------------------------------------------------

----------------------------------------------------------------------------------

### Conclusion

----------------------------------------------------------------------------------

*  In this project, we embarked on a journey to develop a machine learning model for predicting the selling prices of used cars. Initially, we encountered challenges with the Linear Regression Model, as its performance fell short of expectations. However, our exploration led us to the promising avenue of Decision Trees.
  
*  Upon implementing the Decision Tree Regressor (DTR) model, we witnessed a remarkable improvement in predictive accuracy, achieving a score exceeding 85%. This success underscores the importance of exploring alternative algorithms to uncover more nuanced relationships within the data.

*  Further analysis revealed that the Random Forest Regressor (RFR) model outperformed the DTR model, achieving an impressive score of 90%. This advancement highlights the potential of ensemble methods in enhancing model performance.
  
*  Moving forward, we will leverage the robust performance of the RFR model to predict car sales prices with greater confidence and reliability. By harnessing the power of advanced machine learning techniques, we aim to provide valuable insights for buyers and sellers in the dynamic used car market landscape.


```

```

```

```
