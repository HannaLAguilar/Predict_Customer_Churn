# Predict Customer Churn

This is a project for the [Machine Learning DevOps Engineer 
Nanodegree](https://github.com/HannaLAguilar/MLOps_Udacity) from Udacity.

## Project Overview
In this project two machine learning 
models are training to identify if credit card 
customers are most likely to churn the bank.

The file `churn_library_solution.py` contains all the
functions necessary to train and test the model and save the
important images of the results and EDA.

For test and debugging purposes a test file 
`churn_script_logging_adn_test.py` was created with the logging 
functions embedded. 

## Running Files
With python >= 3.7 
- Create an environment and activate it
  ```
  conda create -n churn_ml python=3.7 pandas matplotlib seaborn shap 
  conda activate churn_ml
  ```
- Install sklearn with pip
  ```
  pip install sklearn==0.22
  ```
- In a terminal run `churn_library_solution.py` 
  to train and save the models and images
  ```
  ipython churn_library_solution.py
  ```
- In a terminal run `churn_script_logging_adn_test.py` 
  to test and open the file `churn_library.log` in the
  log folder to check the testing
  ```
  ipython churn_script_logging_adn_test.py
  ```
  You can also use pytest to perform the testing
