# Black-Friday-Sales-Prediction
This project involves building a sales prediction model for Black Friday using a dataset containing information about sales transactions.
The code provided here covers data exploration, analysis, preprocessing, and the development of predictive models using machine learning techniques.

## Prerequisites
Before running the code, make sure you have the following libraries installed:
- Pandas
- Seaborn
- Matplotlib
- Scikit-learn
- NumPy

## Overview of the Code
The code is structured as follows:

1-Loading the Dataset: The code loads the Black Friday sales dataset from a CSV file and provides basic information about the dataset, such as its shape, the first and last samples, and statistical summaries.

2-Exploratory Data Analysis (EDA): The code conducts EDA, including:
- Displaying unique values for each column.
- Visualizing the distribution of values in categorical columns.
- Plotting the distribution of purchase amounts.

3-Bivariate Analysis: The code performs bivariate analysis for various features like 'Occupation,' 'Age,' and 'Gender' against the 'Purchase' variable, providing insights into their relationships.

4-Data Cleaning: Handling missing values by filling them with -2.0 for 'Product_Category_2' and 'Product_Category_3'.

5-Label Encoding: Transforming categorical variables into numerical format using label encoding.

6-Correlation Analysis: Creating a correlation matrix to identify relationships between features.

7-Data Splitting: Splitting the data into input features (X) and the target variable (Y), followed by further splitting into training and testing sets.

8-Model Training and Evaluation:
- Training a Linear Regression model and evaluating its accuracy.
- Training a Decision Tree Regressor model and evaluating its accuracy.
- Training a Random Forest Regressor model and evaluating its accuracy.


## Model Accuracy
The predictive model's accuracy is as follows:
- Linear Regression (LR): 15% accuracy
- Decision Tree Regressor (DT): 55% accuracy
- Random Forest Regressor (RFR): 65% accuracy


## Contribution
Contributions to this project are welcome. If you have suggestions for improving the analysis, optimizing the models, or adding new features to enhance the accuracy of sales prediction, your contributions will be highly appreciated.
