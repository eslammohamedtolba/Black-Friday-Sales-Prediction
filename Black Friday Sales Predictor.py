# Import required dependencies
import pandas as pd
from sklearn.preprocessing import LabelEncoder # label Encoding
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import r2_score
import numpy as np


# Load the dataset
sales_dataset = pd.read_csv('train.csv')
# Show the dataset shape
sales_dataset.shape
# Show the first and last five samples in the dataset
sales_dataset.head()
sales_dataset.tail()
# Find some statistical info about dataset
sales_dataset.describe()


# Show the unique values in each column
for col in sales_dataset.columns:
    print("the column",col,"has",len(sales_dataset[col].unique()),"unique value")

# Show data types info of dataset columns
sales_dataset.info()
# Show count values for all columns except User_ID, Product_ID and Purchase columns and plot its count
for col in sales_dataset.columns:
    if col not in ['User_ID','Product_ID','Purchase']:
        print('the column name called: ',col,"has",sales_dataset[col].value_counts())
        sns.countplot(x=col,data=sales_dataset)
        plt.show()

# plot distribution for User_ID, Product_ID object columns
sns.distplot(sales_dataset['Purchase'],color='red')
plt.show()


# Bivariate analysis for Occupation and Purchase
occupation_plot = sales_dataset.pivot_table(index='Occupation',values='Purchase',aggfunc=np.mean)
occupation_plot.plot(kind='bar',figsize=(10,10))
plt.xlabel('Occupation')
plt.ylabel('Purchase')
plt.title('Occupation and purchase analysis')
plt.xticks(rotation=0)
plt.show()
# Bivariate analysis for Age and Purchase
age_plot = sales_dataset.pivot_table(index='Age',values='Purchase',aggfunc=np.mean)
age_plot.plot(kind='bar',figsize=(10,10))
plt.xlabel('age')
plt.ylabel('Purchase')
plt.title('age and purchase analysis')
plt.xticks(rotation=0)
plt.show()
# Bivariate analysis for Gender and Purchase
Gender_plot = sales_dataset.pivot_table(index='Gender',values='Purchase',aggfunc=np.mean)
Gender_plot.plot(kind='bar',figsize=(10,10))
plt.xlabel('Gender')
plt.ylabel('Purchase')
plt.title('Gender and purchase analysis')
plt.xticks(rotation=0)
plt.show()


# Check if there is any none(missing) values in the dataset to decide if will make a data cleaning or not
sales_dataset.isnull().sum()
# Data cleaning by removing the Product_Category_2 and Product_Category_3 columns
sales_dataset['Product_Category_2'].fillna(-2.0,inplace=True)
sales_dataset['Product_Category_3'].fillna(-2.0,inplace=True)
sales_dataset.head()

# Label encoding
label = LabelEncoder()
sales_dataset.replace({'Age':{'26-35':'30','36-45':'40','18-25':'20','46-50':'50','51-55':'55','55+':'56','0-17':'15'}},inplace=True)
sales_dataset.replace({'Gender':{'M':1,'F':0}},inplace=True)
sales_dataset.replace({'City_Category':{'A':0,'B':1,'C':2}},inplace=True)
sales_dataset.replace({'Stay_In_Current_City_Years':{'4+':4}},inplace=True)


# Create Correlation between all dataset features
correlation_values = sales_dataset.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation_values,square=True,cbar=True,fmt='.1f',annot=True,annot_kws={'size':8},cmap="Blues")
plt.show()
# Split data into input and label data
X = sales_dataset.drop(columns=['Purchase','User_ID','Product_ID'],axis=1)
Y = sales_dataset['Purchase']
print(X.shape,Y.shape)

# Split data into train and test data
x_train,x_test,y_train,y_test = train_test_split(X,Y,train_size=0.8,random_state=42)
print(X.shape,x_train.shape,x_test.shape)
print(Y.shape,y_train.shape,y_test.shape)


# Train the model and make it predict on train and test input data
def train_predict(model,x_train,y_train,x_test,y_test):
    # Train model
    model.fit(x_train,y_train)
    # Make model predict on train and test input data
    predicted_y_train = model.predict(x_train)
    predicted_y_test = model.predict(x_test)
    # Evaluate model
    accuracy_value_train = r2_score(y_train,predicted_y_train)
    accuracy_value_test = r2_score(y_test,predicted_y_test)
    # Return the prediction accuracy on train and test data
    return [accuracy_value_train,accuracy_value_test]


# Create linear regressor model
LRModel = LinearRegression()
accuracy_train,accuracy_test = train_predict(LRModel,x_train,y_train,x_test,y_test)
print(accuracy_train,accuracy_test) # 15 15
# Create Decision Tree Regressor model
DTModel = DecisionTreeRegressor()
accuracy_train,accuracy_test = train_predict(DTModel,x_train,y_train,x_test,y_test)
print(accuracy_train,accuracy_test) # 80 55
# Create Random Forest Regressor model
RandomForestRegressorModel = RandomForestRegressor(n_estimators=150,max_depth=8, random_state=42)
accuracy_train,accuracy_test = train_predict(RandomForestRegressorModel,x_train,y_train,x_test,y_test)
print(accuracy_train,accuracy_test) # 65 65



