# Mini-project IV

### [Assignment](assignment.md)

## Project/Goals
The goal of this project was: 
1. To create a machine learning model that predicts whether an individual will be granted a loan;
2. To create an API using Flask where a post request can be used to determine whether an individual is likely to receive a loan; and
3. To deploy the API to the cloud using AWS and query it to get results using Bash, Postman, or Python (I used Python)

Overall, the completed project workflow is shown in the figures below. 

![](/images/figure1.png)

![](/images/figure2.png)

## Hypothesis
Hypotheses examined included: 
- Applicants who are male are more likely to get loans
- Applicants who are married applicants are more likely to get loans
- Applicants who have graduated have a higher income than applicants who have not graduated. 

These could be tested by comparing the means of the two data sets and performing hypothesis testing using the Kolmogorov-Smirnov test.

## EDA 
EDA process can be found [here](https://github.com/mnicnielsen/deployment-project/blob/master/notebooks/main.ipynb). Notable findings from the EDA processes included: 

- Null values were found in the following columns: 

    - gender
    - married
    - dependents
    - self-employed
    - loan amoun
    - loan amount term
    - credit history

- All variables involving money (applicant income, coapplicant income, and loan amount) were right-skewed. The log of these values was a much more normal distribution; see example below. 

![](/images/figure3.png)


- Applicants who had graduated tended to have higher income than applicants who had not graduated. 

![](/images/figure4.png)

```python
#do hypothesis testing to determine if they are from the same distribution
from scipy.stats import ttest_ind

stat, p_value = ttest_ind(graduates['AnnualIncome'], not_graduates['AnnualIncome'])
print(f"t-test: statistic={stat:.4f}, p-value={p_value:.4f}")
```
- Pivot tables were created to understand the percentage of loans that were approved in different groups. 

| Gender | % Not Approved | % Approved |
| --- | --- | --- |
|Female|33|67|
|Male|31|69|
|Overall|31|69|

Loan amount was correlated with applicant income. 

![](/images/figure5.png)


## Process
All code for this project can be found [here](https://github.com/mnicnielsen/deployment-project/blob/master/notebooks/main.ipynb).

The overall process was as follows: 

1. Performed EDA 
- Described above
2. Performed data cleaning
- imputed modes for null values for categorical variables
- imputed means for null values for numerical variables (likely should have used median as distribution was not normal)
3. Completed additional feature engineering
- transformed money variables using PowerTransformer (to make more normal)
4. Created model through use of pipeline and GridSearchCV. 
- Tried using RandomForestClassifier and XGBoost
- RandomForest Classifier had a better R2 (~81%), so this model was used. 
    - Hyperparameters were optimized using the following code: 
    
    ```python
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=5, shuffle=True, random_state=33)
    params = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt', 'log2'],
        'max_depth' : [4,5,6,7,8],
        'criterion' :['gini', 'entropy']
    }

    clf_gridsearch = GridSearchCV(clf, params, cv=kf)
    clf_gridsearch.fit(X_train, y_train)

    print(clf_gridsearch.best_params_, clf_gridsearch.best_score_)
    ```
    Hyperparameters used were: 
    |Parameter|Value Used|
    |---|---|
    |Criterion|gini|
    |Max_depth|4|
    |Max_features|sqrt|
    |N_estimators|500|
    
5. Combined steps 2-4 Created pipeline using sklearn. 
```python
#define preprocessing steps for pipeline
cat_transform = Pipeline([("impute_mode", SimpleImputer(strategy='most_frequent')), ("one-hot-encode", OneHotEncoder(sparse=False))])
term_transform = Pipeline([("impute_mean", SimpleImputer(strategy='mean')),("scaling", StandardScaler())])
dollar_transform = Pipeline([("impute_mean", SimpleImputer(strategy='mean')),("log_transform", PowerTransformer())])

preprocessing = ColumnTransformer([("cat_transform", cat_transform, ['Gender','Married','Dependents','Education','Self_Employed','Credit_History','Property_Area']),
                                  ("term_transform", term_transform, ['Loan_Amount_Term']),
                                   ("dollar_transform", dollar_transform, ['ApplicantIncome','CoapplicantIncome','LoanAmount']),
                                  ])

#define model step for pipeline
model = RandomForestClassifier(criterion='gini', max_depth=4, max_features='sqrt', n_estimators=500)

#combine steps into pipeline
pipeline = Pipeline([("preprocessing", preprocessing),
                     ("model", model)])
```
Resulting pipeline below:
![](/images/figure6.png)

6. Tested the pipeline
- resulting r^2: 0.8162162162162162

7. Pickled the model.
```python
import pickle
model_columns = list(X.columns)
with open('../data/model_columns.pkl', 'wb') as file:
    pickle.dump(model_columns, file)
    
pickle.dump(pipeline, open('../data/pipeline.pkl', 'wb'))
```
7. Used Flask to deploy the model locally as an API. 
- Created app.py file [here](https://github.com/mnicnielsen/deployment-project/blob/master/notebooks/app.py).
- Started the app using the terminal
- Created python script for post request in Jupyter notebook. 
```## Python test file for flask to test locally
import requests as r
import pandas as pd
import json
import json


base_url = 'http://127.0.0.1:5000/' #base url local host

json_data = {
        "Gender": 'Male',
        "Married": 'Yes',
        "Dependents": '0',
        "Education": 'Graduate',
        "Self_Employed": 'No',
        "ApplicantIncome": 100000,
        "CoapplicantIncome": 0,
        "LoanAmount": 20,
        "Loan_Amount_Term": 120,
        "Credit_History": 1,
        "Property_Area": 'Rural'
        }

# Get Response
# response = r.get(base_url)
response = r.post(base_url + "predict", json = json_data)


if response.status_code == 200:
    print('...')
    print('request successful')
    print('...')
    print(response.json())
else:
    print(response.json())
    print('request failed')
```
8. Deployed to the cloud (AWS instance). 
- Created an app.py file in an AWS instance. 
- Launched using terminal inside of AWS instance.
- Used a jupyter notebook inside the AWS instance to create a post request. 

## Results/Demo
- model's r^2: 0.816
- the API gave the user different messages with the "/" extension and the "/predict" extension. Examples are shown below. 

![](/images/figure7.png)

![](/images/figure8.png)

- a sample result for someone *likely* to get a loan is shown below 

![](/images/figure9.png)

## Challanges 
- Flask App
    - I faced significant challenges getting the flask app up and running for the first time! Figuring out how to properly format the JSON and use the pipeline within the API was challenging. 
- AWS Instance
    - When I copied my code into a file in AWS, I had issues with it running. The issue, I finally figured out, was that the version of sckit-learn on my local machine and in the AWS instance were different, so the pickled version couldn't be unpickled properly, and the model was not able to transform the data. With mentor help, I was able to install a newer version of sckit-learn on my local machine and run the API in the cloud. 

## Future Goals
With more time, I would like to:

- Further assess the accuracy of this model using a confusion matrix, to ensure it's not predicting loan approvals too much of the time.  
- Create a user interface for the Flask App. 