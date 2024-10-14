# %%
import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv(r'C:\Users\neera\Downloads\Householdconsumption\household_power_consumption.csv', sep=',', na_values='?')

# Convert Date and Time to a single datetime column
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.set_index('Datetime', inplace=True)

# Create a new feature: total energy consumption from sub-metering values
df['total_energy'] = df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']

# Select features and target variable
X = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'total_energy']]
y = df['Global_active_power']  # or whichever variable you're targeting

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Cross-validation
lin_cv_rmse = np.sqrt(-cross_val_score(lin_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=5))
print(f'Linear Regression: \n Cross-validated RMSE: {lin_cv_rmse.mean()}')

# Predictions and Evaluation
y_pred = lin_reg.predict(X_test)
lin_test_mse = mean_squared_error(y_test, y_pred)
lin_r2 = r2_score(y_test, y_pred)
print(f'Test MSE: {lin_test_mse}, R^2: {lin_r2}')


# %%
print("Missing values in each column:")
print(X_train.isnull().sum())


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv(r'C:\Users\neera\Downloads\Householdconsumption\household_power_consumption.csv', sep=',', na_values='?')

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'total_energy']] = imputer.fit_transform(
    df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'total_energy']]
)

# Prepare features and target variable (ensure 'total_energy' is your target)
X = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']]
y = df['total_energy']  # Replace with your actual target variable

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Continue with your model evaluations...


# %%
import pandas as pd

# Load the dataset
df = pd.read_csv(r'C:\Users\neera\Downloads\Householdconsumption\household_power_consumption.csv', sep=',', na_values='?')

# Check the columns in the DataFrame
print("Columns in the DataFrame:")
print(df.columns)

# Check the first few rows of the DataFrame
print("\nFirst few rows of the DataFrame:")
print(df.head())


# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

# Load the dataset
df = pd.read_csv(r'C:\Users\neera\Downloads\Householdconsumption\household_power_consumption.csv', sep=',', na_values='?')

# Create the total_energy column based on sub-metering
df['total_energy'] = df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'total_energy']] = imputer.fit_transform(
    df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'total_energy']]
)

# Prepare features and target variable
X = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']]
y = df['total_energy']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# Cross-validation
lin_cv_rmse = np.sqrt(-cross_val_score(lin_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=5))

# Print results
print(f'Linear Regression - Cross-validated RMSE: {lin_cv_rmse.mean()}')


# %%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Random Forest Regression
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)

# Predictions and evaluation
y_pred = rf_reg.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f'Random Forest - RMSE: {rf_rmse}')


# %%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

print(f'Best Params: {grid_search.best_params_}')
best_rf_model = grid_search.best_estimator_


# %%
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize the RandomForestRegressor
rf = RandomForestRegressor()

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Output the best parameters
print(f'Best Params: {grid_search.best_params_}')

# Use the best model
best_rf_model = grid_search.best_estimator_


# %%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import numpy as np

# Load the dataset
df = pd.read_csv(r'C:\Users\neera\Downloads\Householdconsumption\household_power_consumption.csv', sep=',', na_values='?')

# Drop any rows with missing 'Global_active_power'
df.dropna(subset=['Global_active_power'], inplace=True)

# Impute missing values using mean for relevant columns
imputer = SimpleImputer(strategy='mean')
df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']] = imputer.fit_transform(
    df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']]
)

# Create the total energy feature (ensure it's calculated correctly)
df['total_energy'] = df['Global_active_power'] * 1000 / 60  # Example for total energy consumption

# Prepare features (X) and target (y)
X = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity']]
y = df['total_energy']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid for the Random Forest model
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Initialize the RandomForestRegressor
rf = RandomForestRegressor()

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Output the best parameters
print(f'Best Params: {grid_search.best_params_}')

# Use the best model
best_rf_model = grid_search.best_estimator_


# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r'C:\Users\neera\Downloads\Householdconsumption\household_power_consumption.csv', sep=',', na_values='?')

# Rename columns for easier access
data.columns = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage',
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

# Combine 'Date' and 'Time' into a single datetime column
data['Datetime'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')

# Convert columns to numeric, ignoring non-numeric data
for col in ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop 'Date' and 'Time' columns
data.drop(columns=['Date', 'Time'], inplace=True)

# Create a new feature 'total_energy'
data['total_energy'] = (data['Global_active_power'] * 1000) / 60.0

# Drop rows with missing values
data.dropna(inplace=True)

# Split data into training and testing sets (80/20)
X = data[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 
          'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]
y = data['total_energy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
import pandas as pd

# Load the dataset
df = pd.read_csv(r'C:\Users\neera\Downloads\Householdconsumption\household_power_consumption.csv', sep=',', na_values='?')

# Rename columns for easier access
df.columns = ['Date', 'Time', 'Global_active_power', 'Global_reactive_power', 'Voltage',
                'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']

# Combine 'Date' and 'Time' into a single datetime column
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')

# Convert columns to numeric, ignoring non-numeric data
for col in ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop 'Date' and 'Time' columns
df.drop(columns=['Date', 'Time'], inplace=True)

# Create a new feature 'total_energy'
df['total_energy'] = (df['Global_active_power'] * 1000) / 60.0

# Drop rows with missing values
df.dropna(inplace=True)

# Display the first few rows
df.head()


# %%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Select features and target variable
X = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 
        'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']]
y = df['total_energy']

# Split the dataset into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Random Forest Regressor and hyperparameter grid
rf = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Use GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

# Get the best parameters
best_rf = grid_search.best_estimator_

# Predict on the test set
y_pred = best_rf.predict(X_test)

# Evaluate the model performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')

# Plot feature importance
importances = best_rf.feature_importances_
features = X.columns
indices = importances.argsort()[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[indices], y=features[indices])
plt.title('Feature Importance')
plt.show()

# Plot actual vs predicted values
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Total Energy Consumption')
plt.xlabel('Instance')
plt.ylabel('Total Energy Consumption')
plt.show()


# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from sklearn.inspection import permutation_importance

# Load dataset
df = pd.read_csv(r'C:\Users\neera\Downloads\Householdconsumption\household_power_consumption.csv', sep=',', na_values='?')  # Adjust the file path as needed

# Preprocessing
df['Total_energy'] = (df['Global_active_power'] * 1000) / 60
df = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'Total_energy']]
df.dropna(inplace=True)

# Splitting the dataset
X = df.drop('Total_energy', axis=1)
y = df['Total_energy']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
rf_model = RandomForestRegressor(n_jobs=-1, random_state=42)

# Hyperparameter tuning with parallel processing
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Optimal parameters
best_params = grid_search.best_params_
print(f'Optimal Parameters: {best_params}')

# Evaluate the model
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Performance Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f'Test MSE: {mse}, Test MAE: {mae}')

# Visualizations
# Feature Importance
importances = best_model.feature_importances_
features = X.columns
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Feature Importances')
plt.bar(range(X.shape[1]), importances[indices], align='center')
plt.xticks(range(X.shape[1]), features[indices], rotation=45)
plt.xlim([-1, X.shape[1]])
plt.ylabel('Importance')
plt.show()

# Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual Total Energy Consumption')
plt.ylabel('Predicted Total Energy Consumption')
plt.title('Actual vs Predicted')
plt.show()

# Residual Plot
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='k', linestyle='--')
plt.xlabel('Predicted Total Energy Consumption')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.show()

# Permutation Importance
perm_importance = permutation_importance(best_model, X_test, y_test, n_repeats=30, random_state=42, n_jobs=-1)

# Plot Permutation Importance
sorted_idx = perm_importance.importances_mean.argsort()

plt.figure(figsize=(10, 6))
plt.boxplot(perm_importance.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
plt.title('Permutation Importances (test set)')
plt.show()


# %%


# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Sample Data (Replace with your dataset)
# Assuming you have data in variables `X` (features) and `y` (target)
# Replace this with your actual dataset loading/preprocessing
# Example: df = pd.read_csv("your_data.csv")
# X = df.drop(columns=['target'])
# y = df['target']
X, y = np.random.rand(100, 10), np.random.rand(100)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize RandomForestRegressor
rf_model = RandomForestRegressor(random_state=42)

# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# GridSearchCV with n_jobs=1 to avoid parallel processing issues
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                           scoring='neg_mean_squared_error', cv=5, n_jobs=1)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the optimal parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Make predictions using the best estimator
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Calculate and print the mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# %%
pip install scikit-learn statsmodels tensorflow bayesian-optimization


# %%
pip install FuzzyTM
pip install blosc2
pip install cython


# %%
pip install numpy==1.24.0


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import statsmodels.api as sm
from bayes_opt import BayesianOptimization

# Load dataset
df = pd.read_csv(r'C:\Users\neera\Downloads\Householdconsumption\household_power_consumption.csv', sep=',', na_values='?')

# Preprocessing: Combine Date and Time columns into a single 'Datetime' column
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.set_index('Datetime', inplace=True)

# Feature Engineering: Create a 'total_energy' feature
df['total_energy'] = df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']

# Select features (X) and target variable (y)
X = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'total_energy']]
y = df['Global_active_power']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------- 1. Gradient Boosting Regressor -------
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
print(f'Gradient Boosting MSE: {mse_gbr}')

# ------- 2. ARIMA Model -------
# Aggregate data for ARIMA model by averaging over 1-hour intervals
df_arima = df.resample('H').mean()
arima_model = sm.tsa.ARIMA(df_arima['Global_active_power'].dropna(), order=(5,1,0))
arima_result = arima_model.fit()
y_pred_arima = arima_result.forecast(steps=len(X_test))[0]
mse_arima = mean_squared_error(y_test, y_pred_arima)
print(f'ARIMA MSE: {mse_arima}')

# ------- 3. RandomizedSearchCV for Random Forest -------
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rfr = RandomForestRegressor(random_state=42)
rand_search_rf = RandomizedSearchCV(rfr, param_distributions=param_dist, n_iter=10, cv=5, random_state=42)
rand_search_rf.fit(X_train, y_train)
y_pred_rf_rand = rand_search_rf.predict(X_test)
mse_rf_rand = mean_squared_error(y_test, y_pred_rf_rand)
print(f'Random Forest (RandomizedSearchCV) MSE: {mse_rf_rand}')

# ------- 4. Bayesian Optimization for Gradient Boosting -------
def gbr_evaluate(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    params = {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'min_samples_split': int(min_samples_split),
        'min_samples_leaf': int(min_samples_leaf)
    }
    model = GradientBoostingRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return -mean_squared_error(y_test, pred)

gbr_bayes_opt = BayesianOptimization(
    f=gbr_evaluate,
    pbounds={
        'n_estimators': (50, 200),
        'max_depth': (3, 20),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 4)
    },
    random_state=42
)
gbr_bayes_opt.maximize(init_points=2, n_iter=10)

best_params_gbr_bayes = gbr_bayes_opt.max['params']
gbr_bayes_model = GradientBoostingRegressor(
    n_estimators=int(best_params_gbr_bayes['n_estimators']),
    max_depth=int(best_params_gbr_bayes['max_depth']),
    min_samples_split=int(best_params_gbr_bayes['min_samples_split']),
    min_samples_leaf=int(best_params_gbr_bayes['min_samples_leaf']),
    random_state=42
)
gbr_bayes_model.fit(X_train, y_train)
y_pred_gbr_bayes = gbr_bayes_model.predict(X_test)
mse_gbr_bayes = mean_squared_error(y_test, y_pred_gbr_bayes)
print(f'Gradient Boosting (Bayesian Opt) MSE: {mse_gbr_bayes}')

# ------- Cross-referencing all results -------
results = {
    'MSE_Gradient_Boosting': mse_gbr,
    'MSE_ARIMA': mse_arima,
    'MSE_Random_Forest_RandomSearch': mse_rf_rand,
    'MSE_Gradient_Boosting_Bayesian_Opt': mse_gbr_bayes
}

print("\nCross-referenced Results:")
for model, mse in results.items():
    print(f'{model}: {mse}')


# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
import statsmodels.api as sm
from bayes_opt import BayesianOptimization

# Load dataset
df = pd.read_csv(r'C:\Users\neera\Downloads\Householdconsumption\household_power_consumption.csv', sep=',', na_values='?')

# Preprocessing: Combine Date and Time columns into a single 'Datetime' column
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S')
df.set_index('Datetime', inplace=True)

# Feature Engineering: Create a 'total_energy' feature
df['total_energy'] = df['Sub_metering_1'] + df['Sub_metering_2'] + df['Sub_metering_3']

# Select features (X) and target variable (y)
X = df[['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity', 'total_energy']]
y = df['Global_active_power']

# Handle NaN values by filling them with the mean of each column
X.fillna(X.mean(), inplace=True)
y.fillna(y.mean(), inplace=True)  # Ensure target variable also has no NaN values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------- 1. Gradient Boosting Regressor -------
gbr = GradientBoostingRegressor(random_state=42)
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
print(f'Gradient Boosting MSE: {mse_gbr}')

# ------- 2. ARIMA Model -------
# Aggregate data for ARIMA model by averaging over 1-hour intervals
df_arima = df.resample('H').mean()
arima_model = sm.tsa.ARIMA(df_arima['Global_active_power'].dropna(), order=(5, 1, 0))
arima_result = arima_model.fit()
y_pred_arima = arima_result.forecast(steps=len(X_test))
mse_arima = mean_squared_error(y_test, y_pred_arima)
print(f'ARIMA MSE: {mse_arima}')

# ------- 3. RandomizedSearchCV for Random Forest -------
param_dist = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rfr = RandomForestRegressor(random_state=42)
rand_search_rf = RandomizedSearchCV(rfr, param_distributions=param_dist, n_iter=10, cv=5, random_state=42)
rand_search_rf.fit(X_train, y_train)
y_pred_rf_rand = rand_search_rf.predict(X_test)
mse_rf_rand = mean_squared_error(y_test, y_pred_rf_rand)
print(f'Random Forest (RandomizedSearchCV) MSE: {mse_rf_rand}')

# ------- 4. Bayesian Optimization for Gradient Boosting -------
def gbr_evaluate(n_estimators, max_depth, min_samples_split, min_samples_leaf):
    params = {
        'n_estimators': int(n_estimators),
        'max_depth': int(max_depth),
        'min_samples_split': int(min_samples_split),
        'min_samples_leaf': int(min_samples_leaf)
    }
    model = GradientBoostingRegressor(**params, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return -mean_squared_error(y_test, pred)

gbr_bayes_opt = BayesianOptimization(
    f=gbr_evaluate,
    pbounds={
        'n_estimators': (50, 200),
        'max_depth': (3, 20),
        'min_samples_split': (2, 10),
        'min_samples_leaf': (1, 4)
    },
    random_state=42
)
gbr_bayes_opt.maximize(init_points=2, n_iter=10)

best_params_gbr_bayes = gbr_bayes_opt.max['params']
gbr_bayes_model = GradientBoostingRegressor(
    n_estimators=int(best_params_gbr_bayes['n_estimators']),
    max_depth=int(best_params_gbr_bayes['max_depth']),
    min_samples_split=int(best_params_gbr_bayes['min_samples_split']),
    min_samples_leaf=int(best_params_gbr_bayes['min_samples_leaf']),
    random_state=42
)
gbr_bayes_model.fit(X_train, y_train)
y_pred_gbr_bayes = gbr_bayes_model.predict(X_test)
mse_gbr_bayes = mean_squared_error(y_test, y_pred_gbr_bayes)
print(f'Gradient Boosting (Bayesian Opt) MSE: {mse_gbr_bayes}')

# ------- Cross-referencing all results -------
results = {
    'MSE_Gradient_Boosting': mse_gbr,
    'MSE_ARIMA': mse_arima,
    'MSE_Random_Forest_RandomSearch': mse_rf_rand,
    'MSE_Gradient_Boosting_Bayesian_Opt': mse_gbr_bayes
}

print("\nCross-referenced Results:")
for model, mse in results.items():
    print(f'{model}: {mse}')


# %%
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Load your data
# df = pd.read_csv('your_data.csv')  # Uncomment and provide your file path
# Assuming df is already defined as per your dataset structure

# Data Cleaning
# Convert date columns to datetime
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Fill missing values in numerical columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    df[col].fillna(df[col].mean(), inplace=True)

# Fill missing values in non-numeric columns (if applicable)
df.fillna(method='ffill', inplace=True)  # Forward fill for categorical data

# Check if 'Global_active_power' is numeric
if 'Global_active_power' in df.columns:
    df['Global_active_power'] = pd.to_numeric(df['Global_active_power'], errors='coerce')

# Drop rows with NaN values after conversions
df.dropna(inplace=True)

# Feature Selection
X = df.drop(columns=['Global_active_power'])  # Drop target column
y = df['Global_active_power']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------- 1. Gradient Boosting Model -------
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)
y_pred_gbr = gbr.predict(X_test)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
print(f'Gradient Boosting MSE: {mse_gbr}')

# ------- 2. ARIMA Model -------
# Aggregate data for ARIMA model by averaging over 1-hour intervals
df_arima = df.set_index('Date').resample('H').mean()
arima_model = sm.tsa.ARIMA(df_arima['Global_active_power'].dropna(), order=(5, 1, 0))
arima_result = arima_model.fit()

# Print ARIMA model summary
print(arima_result.summary())


# %%
# Convert the 'Date' column to datetime and extract features
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day

# Drop the original Date column if it's no longer needed
df.drop(columns=['Date'], inplace=True)


# %%
print(df.dtypes)


# %%
df.dropna(inplace=True)  # This drops any rows with NaN values


# %%
y = df['your_target_column']  # Replace with your actual target column
print(y.dtypes)


# %%
# Set the target variable
y = df['total_energy']  # This should match the name of your target column

# Set the features (drop the target variable from the DataFrame)
X = df.drop(columns=['total_energy'])  # Drop the target variable from features

# Check the types of X and y
print(X.dtypes)
print(y.dtypes)


# %%
df['Time'] = pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce')
df['Hour'] = df['Time'].dt.hour
df['Minute'] = df['Time'].dt.minute
df['Second'] = df['Time'].dt.second
df.drop(columns=['Time'], inplace=True)
df.dropna(inplace=True)  # Drop rows with NaN values


# %%
y = df['total_energy']  # Your target variable
X = df.drop(columns=['total_energy'])  # Features


# %%
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the Gradient Boosting model
gbr = GradientBoostingRegressor()
gbr.fit(X_train, y_train)

# Make predictions and evaluate
y_pred_gbr = gbr.predict(X_test)
mse_gbr = mean_squared_error(y_test, y_pred_gbr)
print(f'Gradient Boosting MSE: {mse_gbr}')


# %%
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_gbr)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Identity line
plt.xlabel('Actual Total Energy')
plt.ylabel('Predicted Total Energy')
plt.title('Actual vs Predicted Total Energy')
plt.show()


# %%
import numpy as np

feature_importances = gbr.feature_importances_
sorted_idx = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(X.shape[1]), feature_importances[sorted_idx], align='center')
plt.xticks(range(X.shape[1]), X.columns[sorted_idx], rotation=90)
plt.title('Feature Importances')
plt.show()


# %%
# Ensure 'Date' is your datetime index and drop any missing values
df.set_index('Date', inplace=True)
df = df['total_energy'].dropna()


# %%
train_size = int(len(df) * 0.8)
train, test = df[:train_size], df[train_size:]


# %%
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Example parameters: p=1, d=1, q=1 (you may need to adjust these)
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

# Print model summary
print(model_fit.summary())


# %%
train = df['total_energy']  # Replace 'total_energy' with your target column name


# %%
train.index = df['Date']  # Assuming 'Date' is your datetime column


# %%
print(df.columns)


# %%
import pandas as pd

# Assuming your DataFrame is named df
# Create a 'Date' column
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute', 'Second']])

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Now you can use the 'total_energy' column as your target
train = df['total_energy']

# Fit the ARIMA model
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

# Print model summary
print(model_fit.summary())


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Calculate residuals
residuals = model_fit.resid

# Residuals plot
plt.figure(figsize=(12, 6))

# Plot residuals
plt.subplot(2, 2, 1)
plt.plot(residuals)
plt.title('Residuals')
plt.axhline(0, color='red', linestyle='--')

# Histogram of residuals
plt.subplot(2, 2, 2)
sns.histplot(residuals, kde=True)
plt.title('Histogram of Residuals')

# Q-Q plot
plt.subplot(2, 2, 3)
sm.qqplot(residuals, line='s', ax=plt.gca())
plt.title('Q-Q Plot')

# Residuals vs Fitted
plt.subplot(2, 2, 4)
plt.scatter(model_fit.fittedvalues, residuals)
plt.axhline(0, color='red', linestyle='--')
plt.title('Residuals vs Fitted')

plt.tight_layout()
plt.show()


# %%
# Forecasting future values
n_forecast = 30  # Change this to how many steps you want to forecast
forecast = model_fit.forecast(steps=n_forecast)

# Create a date range for the forecast
last_date = df['Date'].iloc[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_forecast)

# Create a DataFrame for the forecasted values
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecasted_Total_Energy'])

print(forecast_df)


# %%
print(df.columns)


# %%
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Create the 'Date' column
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])

# Set the 'Date' column as the index
df.set_index('Date', inplace=True)

# Ensure you have the correct target variable
train = df['total_energy']  # Using 'total_energy' as the target variable

# Fit the ARIMA model
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

# Print model summary
print(model_fit.summary())

# Forecasting future values
n_forecast = 30  # Number of steps to forecast
forecast = model_fit.forecast(steps=n_forecast)

# Create a date range for the forecast
last_date = df.index[-1]
forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_forecast)

# Create a DataFrame for the forecasted values
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['Forecasted_Total_Energy'])

# Visualize the actual vs. forecasted values
plt.figure(figsize=(12, 6))
plt.plot(train, label='Actual Total Energy', color='blue')
plt.plot(forecast_df, label='Forecasted Total Energy', color='orange')
plt.title('Total Energy Forecast')
plt.xlabel('Date')
plt.ylabel('Total Energy')
plt.legend()
plt.show()


# %%
# Get residuals
residuals = model_fit.resid

# Plot the residuals
plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.plot(residuals)
plt.title('Residuals of ARIMA Model')
plt.xlabel('Date')
plt.ylabel('Residuals')
plt.axhline(0, color='red', linestyle='--')

plt.subplot(212)
plt.hist(residuals, bins=30, edgecolor='black')
plt.title('Histogram of Residuals')
plt.xlabel('Residuals')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Perform the Shapiro-Wilk test for normality
from scipy.stats import shapiro
stat, p = shapiro(residuals)
print(f'Statistic: {stat}, p-value: {p}')


# %%
import statsmodels.api as sm

# Q-Q plot
plt.figure(figsize=(8, 6))
sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.show()


# %%
import statsmodels.api as sm

# Q-Q plot
plt.figure(figsize=(8, 6))
sm.qqplot(residuals, line='s')
plt.title('Q-Q Plot of Residuals')
plt.show()



