import pandas as pd
from sklearn.preprocessing import MinMaxScaler
unseen_df_pd = pd.read_csv("C:/Users/YashRaj/AppData/Local/Programs/Python/Python310/attrition_prediction/df_unseen.csv")

unseen_df_pd2 = unseen_df_pd[['Age', 'Gender', 'MaritalStatus', 'MonthlyIncome',
       'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
       'YearsInCurrentRole', 'YearsSinceLastPromotion',
       'YearsWithCurrManager']]

# Making categorical variables into numeric representation
 
new_unseen_df_pd = pd.get_dummies(unseen_df_pd2, columns = ['MaritalStatus', 'Gender'])

# Scaling our columns
 
scale_vars = ['MonthlyIncome']
scaler = MinMaxScaler()
new_unseen_df_pd[scale_vars] = scaler.fit_transform(new_unseen_df_pd[scale_vars])

# Loading our model

import pickle

dt = pickle.load(open('C:/Users/YashRaj/AppData/Local/Programs/Python/Python310/attrition_prediction/model.pkl', 'rb'))
pred_decitree = dt.predict(new_unseen_df_pd.values)
pred_prob_decitree = dt.predict_proba(new_unseen_df_pd.values)

output = unseen_df_pd.copy()
output['Predictions - Attrition or Not'] = pred_decitree
