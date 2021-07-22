#!/usr/bin/env python
# coding: utf-8

# In[14]:


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib
import matplotlib.pyplot as plt
# get_ipython().run_line_magic('matplotlib', 'inline')
import pandas

import os

matplotlib.rc('font', size=16)
matplotlib.style.use('ggplot')


# ### Problem: Boston Housing Dataset

# #### For reference, here is the description key for the features:
# 
# 
# |feature name | description|
# |-------------|------------|
# |CRIM| Per capita crime rate by town|
# |ZN| Proportion of residential land zoned for lots over 25,000 sq. ft|
# |INDUS| Proportion of non-retail business acres per town|
# |CHAS| Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)|
# |NOX| Nitric oxide concentration (parts per 10 million)|
# |RM| Average number of rooms per dwelling|
# |AGE| Proportion of owner-occupied units built prior to 1940|
# |DIS| Weighted distances to five Boston employment centers|
# |RAD| Index of accessibility to radial highways|
# |TAX| Full-value property tax rate per 10,000 dollar|
# |PTRATIO| Pupil-teacher ratio by town|
# |B| 1000(Bk — 0.63)², where Bk is the proportion of [people of African American descent] by town|
# |LSTAT| Percentage of lower status of the population|
# |MEDV| Median value of owner-occupied homes in $1000s|
# 

# ### 1. Import the file `boston_housing.csv` from the `data` directory
# * Refer to the imported data as `df`
# * Review the feature names and their datatypes
# * Display the first 5 rows of the dataframe

# In[15]:


filename = 'boston_housing.csv'

df = pd.read_csv(filename)
# df.info()


# In[16]:


df.head(5)


# ### 2. Use Seaborn's `heatmap` to view correlation coefficients between the indicated subset of the dataset's features
# * Note that all features are numeric
# * Save the correlation coefficients to these respective variable names:
#     * `rm_rad`
#     * `ptratio_mdev`
#     * `chas_nox`

# In[17]:


feature_subset = ['CHAS', 'RM', 'LSTAT', 'RAD', 'NOX', 'PTRATIO', 'MDEV']

sns.heatmap(df[feature_subset].corr(), 
            annot=True)

rm_rad = -0.21
ptratio_mdev = -0.51
chas_nox = 0.091


# ### 3. Use Seaborn's `pairplot` to graph the numeric features found in the heatmap that have correlation coefficients greater than 0.6 or less than -0.6 in relation to `MDEV` (Median value of owner-occupied homes in 1000s of dollars)
# * We will begin to consider `MDEV` as the target value
# * Store the feature names in a list called `corr_features`

# In[18]:


corr_features = ['RM', 'LSTAT', 'MDEV']

sns.pairplot(df[corr_features], 
             corner=True, kind='reg', markers='+')


# The selected features should be sufficient for applying and interpreting our multiple linear regression.

# ### 4. Represent the target (`MDEV`) with variable `y` , and the features you selected in Question 3 with variable `X_mult`
# * Note: there should be 2 features

# In[19]:


y = df['MDEV'] 
X_mult = df[['RM', 'LSTAT']]


# ### 5. Apply SKLearn's `LinearRegression` model for multiple features
# * refer to the model with a variable named `model_mult`
# * refer to $\beta_0$ in a variable called `b_0_mult`
# * refer to the other betas with a variable called `coefs_mult`

# In[20]:


from sklearn.linear_model import LinearRegression

model_mult = LinearRegression().fit(X_mult, y)

b_0_mult, coefs_mult = model_mult.intercept_, model_mult.coef_

# print(b_0_mult)
# print(coefs_mult[1])


# ### 6. Apply SKLearn's `LinearRegression` model for the single feature `PTRATIO`
# * refer to the input data as `X_single`
# * refer to the model with a variable named `model_single`
# * refer to $\beta_0$ in a variable called `b_0_single`
# * refer to the other betas with a variable called `coefs_single`

# In[21]:


X_single = df[['PTRATIO']]

model_single = LinearRegression().fit(X_single, y)

b_0_single, coefs_single = model_single.intercept_, model_single.coef_

#print(b_0_single)
# print(coefs_single[0])


# ### 7. Compare R-squared and MSE between the multiple model with highly correlated features, and single model with the single, less-correlated feature.
# * Use these variable names to reference predictions
#     * `y_hat_mult`
#     * `y_hat_single`
# * use these variable names to reference measurements
#     * `r2_mult`
#     * `r2_single`
#     * `mse_mult`
#     * `mse_single`
#     * `ssr_mult`
#     * `ssr_single`
# * create a variable called `best_scores`
#     * this will be a tuple in which you put the best R-squared score, the best MSE, then the best ssr, in that order

# In[23]:


from sklearn.metrics import r2_score, mean_squared_error

y_hat_mult = model_mult.predict(X_mult)
y_hat_single = model_single.predict(X_single)

r2_mult = r2_score(y, y_hat_mult)
r2_single = r2_score(y, y_hat_single)

mse_mult = mean_squared_error(y, y_hat_mult)
mse_single = mean_squared_error(y, y_hat_single)

def ssr(y, y_hat):
    resids = y - y_hat
    return (resids**2).sum()

ssr_mult = ssr(y, y_hat_mult)
ssr_single = ssr(y, y_hat_single)

# print(f'Multiple LR R-Squared: {round(r2_mult, 3)}')
# print(f'Single LR R-Squared:   {round(r2_single, 3)}')

# print(f'Multiple LR MSE: {round(mse_mult)}')
# print(f'Single LR MSE: {round(mse_single)}')

# print(f'Multiple LR Sum Squared Residuals: {round(ssr_mult)}')
# print(f'Single LR Sum Squared Residuals: {round(ssr_single)}')

best_scores = (r2_mult, mse_mult, ssr_mult)
# print(isinstance(best_scores, tuple))


# ### For the next few questions, you will use `statsmodels` to answer inferential questions related to Linear Regression applied to the Boston Housing Dataset. 
# * Take a moment to view the pairplot below for the Boston Housing Dataset
#     * Note that not all features are included here, for the sake of having a more easily viewable plot

# In[24]:


sns.pairplot(df[feature_subset], kind='reg', markers='+')


# ### 8. Is an increasing Pupil-teacher ratio (`PTRATIO`) associated with decreasing average rooms per house (`RM`), as might be indicated by the line of best fit in the pairplot?
# * define your formula for `statsmodels`
# * consider `PTRATIO` to be the target
# * view the `summary()` of your OLS model
# * consider a level of significance of 5%
# * set the variable `significant_ptratio_rm` to `True` if you are confident in the above relationship. Otherwise, set the variable to `False`.

# In[25]:


import statsmodels.formula.api as smf

target = 'PTRATIO'
feature = 'RM'

formula = f'{target} ~ {feature}'

model_ptratio_rm = smf.ols(formula=formula, data=df).fit()

# print(model_ptratio_rm.summary())

significant_ptratio_rm = True


# #### Explanation:
# The p-value for the t-test here is close to zero, indicating that there is a relationship between the two features. Note that the `coef` relationship between `RM` and `PTRATIO` is negative. We can thus reject the hypothesis that there is no negative relationship and accept that there is a negative relationship where as `RM` decreases, `PTRATIO` increases.

# ### 9. You suspect that there may be a relationship between decreased accessibility to radial highways (`RAD`) paired with increased Nitric Oxide concentration (`NOX`) toward an increasing percentage of people occupying lower economic status (`LSTAT`). Verify your intuition.
# * define your formula for `statsmodels`
# * consider `LSTAT` to be the target
# * view the `summary()` of your OLS model
# * consider a level of significance of 5%
# * set the variable `significant_lstat_rad_nox` to `True` if you are confident in the above described relationship. Otherwise, set the variable to `False`.
# 

# In[26]:


target = 'LSTAT'
feature_1 = 'NOX'
feature_2 = 'RAD'

# Check paired features to target
formula = f'{target} ~ {feature_1} + {feature_2}'

model_lstat_rad_nox = smf.ols(formula=formula, data=df).fit()

# print(model_lstat_rad_nox.summary())


significant_lstat_rad_nox = False

# # Check features independent of each other: LSTAT ~ NOX
# formula = f'{target} ~ {feature_1}'

# model_lstat_nox = smf.ols(formula=formula, data=df).fit()

# print(model_lstat_nox.summary())


# Check features independent of each other: LSTAT ~ RAD
# formula = f'{target} ~ {feature_2}'

# model_lstat_rad = smf.ols(formula=formula, data=df).fit()

# print(model_lstat_rad.summary())


# #### Explanation:
# 
# The p-value for the t-test here is close to zero for both features to the target. However, As can be seen in the summary, the coefficients for both features are positive. The assumption that increased Nitric oxide concentration coupled and decreaed access to radial highways has a negative relationship with percentage of lower economic status is incorrect. Given the original framing of the question, `significant_lstat_rad_nox` should be set to `False`.

# ### 10. Given the results of your prior inference, you now suspect that there may be a relationship between accessibility to radial highways (`RAD`) and Nitric Oxide concentration (`NOX`). Verify your suspicion.
# * define your formula for `statsmodels`
# * consider `NOX` to be the target
# * view the `summary()` of your OLS model
# * consider a level of significance of 5%
# * set the variable `significant_nox_rad` to `True` if you are confident that there is a  relationship. Otherwise, set the variable to `False`.
# * set the variable `nox_rad_relationship` to `positive` if there is a positive relationship, `negative` if there is a negative relationship, or `none` if you set `significant_nox_rad` to `False`
# 

# In[27]:


import statsmodels.formula.api as smf

target = 'NOX'
feature = 'RAD'

formula = f'{target} ~ {feature}'

model_nox_rad = smf.ols(formula=formula, data=df).fit()

# print(model_nox_rad.summary())

significant_nox_rad = True
nox_rad_relationship = 'positive'


# ### Explanation:
# 
# There is a positive relationship, where increased accessibility to radial highways is associated with increased Nitric Oxide levels. 
# 
# Note: Be careful not to extrapolate further regarding this relationship. For example, proximity to highway onramps or highways themselves may not mean that exhaust is the cause of increased Nitric Oxide levels. 

# ### 11. (Extra Credit!) What feature in the dataset is least likely to have a relationship with all other features aside from the target `MDEV`?
# * consider a signicance level of 5%
# * set the variable `least_related` to the appropriate feature
# 

# In[28]:


# An easy, brute force visual check of summaries in case the heatmap and pairgrid don't give reasonable intuition

for target in df.columns:
    for feature in df.columns:
        if feature == target: continue
#         print('\n\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
#         print(f'\t\t\n{target} ~ {feature}\n')
        formula = f'{target} ~ {feature}'

        model_ptratio_rm = smf.ols(formula=formula, data=df).fit()

#         print(model_ptratio_rm.summary())

least_related = 'CHAS'


# ### Explanation:
# 
# The feature `CHAS` is a binary-encoded indicator of proximity to the Charles river. The p-values for its t-test in relation as a target or feature with all other columns in the data (aside from `MDEV`) exceed the 5% significance level, meaning that `CHAS` can be considered irrelevant in the prediction of any of the other features as targets, aside from the Median value of owner-occupied homes (`MDEV`).

# In[ ]:




