import pandas as pd

data = {'Years': [1989, 1990, 1991, 1992, 1993, 1994, 1995],
        'Boxes': [21, 19.4, 22.6, 28.2, 30.4, 24, 25]}
df = pd.DataFrame(data)
display(df)

from scipy.stats import linregress

slope, intercept, r_value, p_value, std_err = linregress(df['Years'], df['Boxes'])

print(f"Slope: {slope}")
print(f"Intercept: {intercept}")

df['Predicted_Boxes'] = slope * df['Years'] + intercept
df['Percent_of_Trend'] = (df['Boxes'] / df['Predicted_Boxes']) * 100
display(df)

df['Relative_Cyclical_Residual'] = ((df['Boxes'] - df['Predicted_Boxes']) / df['Predicted_Boxes']) * 100
display(df)

max_pct_trend_year = df.loc[df['Percent_of_Trend'].abs().idxmax(), 'Years']
max_rcr_year = df.loc[df['Relative_Cyclical_Residual'].abs().idxmax(), 'Years']

print(f"Year with biggest fluctuation in Percent of Trend: {max_pct_trend_year}")
print(f"Year with biggest fluctuation in Relative Cyclical Residual: {max_rcr_year}")

if max_pct_trend_year == max_rcr_year:
    print("The year is the same for both measures of cyclical variation.")
else:
    print("The year is different for the two measures of cyclical variation.")

column_names = ['Spring', 'Summer', 'Fall', 'Winter']
index_values = [1991, 1992, 1993, 1994, 1995]
data = [[102, 120, 90, 78],
        [110, 126, 95, 83],
        [111, 128, 97, 86],
        [115, 135, 103, 91],
        [122, 144, 110, 98]]
df_seasonal = pd.DataFrame(data, columns=column_names, index=index_values)
display(df_seasonal)

df_unpivoted = df_seasonal.unstack().reset_index()
df_unpivoted.columns = ['Season', 'Year', 'Value']
df_unpivoted = df_unpivoted.sort_values(by='Year')
display(df_unpivoted[['Year', 'Season', 'Value']])
df_unpivoted['4_quarter_moving_total'] = df_unpivoted['Value'].rolling(window=4).sum()
display(df_unpivoted)

df_unpivoted['4_quarter_moving_average'] = df_unpivoted['4_quarter_moving_total'] / 4
display(df_unpivoted)

df_unpivoted['4_quarter_centered_moving_average'] = df_unpivoted['4_quarter_moving_average'].rolling(window=2).mean()
display(df_unpivoted)

df_unpivoted['Percentage_of_Actual_to_Moving_Average'] = (df_unpivoted['Value'] / df_unpivoted['4_quarter_centered_moving_average']) * 100
display(df_unpivoted)

modified_seasonal_indices = df_unpivoted.groupby('Season')['Percentage_of_Actual_to_Moving_Average'].median()
display(modified_seasonal_indices)

average_modified_seasonal_indices = modified_seasonal_indices.mean()
normalization_factor = 400 / average_modified_seasonal_indices
seasonal_indices = modified_seasonal_indices * normalization_factor
display(seasonal_indices)

season = ['Spring', 'Summer', 'Fall', 'Winter']
year = ['1992', '1993', '1994', '1995']

data = [[5.6, 6.8, 6.3, 5.2],
        [5.7, 6.7, 6.4, 5.4],
        [5.3, 6.6, 6.1, 5.1],
        [5.4, 6.9, 6.2, 5.3]]

import pandas as pd
df_stock = pd.DataFrame(data, columns=season, index=year)
display(df_stock)

df_unpivoted_stock = df_stock.unstack().reset_index()
df_unpivoted_stock.columns = ['Quarter', 'Year', 'value']
df_unpivoted_stock = df_unpivoted_stock.sort_values(by='Year')
display(df_unpivoted_stock)

df_unpivoted_stock['4_quarter_moving_total'] = df_unpivoted_stock['value'].rolling(window=4).sum()
df_unpivoted_stock['4_quarter_moving_average'] = df_unpivoted_stock['4_quarter_moving_total'] / 4
display(df_unpivoted_stock)

df_unpivoted_stock['4_quarter_centered_moving_average'] = df_unpivoted_stock['4_quarter_moving_average'].rolling(window=2).mean()
display(df_unpivoted_stock)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(df_unpivoted_stock.index, df_unpivoted_stock['value'], label='Original Data')
plt.plot(df_unpivoted_stock.index, df_unpivoted_stock['4_quarter_centered_moving_average'], label='4-Quarter Centered Moving Average')
plt.xlabel('Time Period')
plt.ylabel('Value')
plt.title('Original Data vs. 4-Quarter Centered Moving Average')
plt.legend()
plt.show()

import pandas as pd
season = ['Winter', 'Spring', 'Summer', 'Fall']
year = ['1992', '1993', '1994', '1995']

df_stock = pd.DataFrame(data, columns=season, index=year)
display(df_stock)

df_unpivoted_stock = df_stock.unstack().reset_index()
df_unpivoted_stock.columns = ['Quarter', 'Year', 'value']
df_unpivoted_stock=df_unpivoted_stock.sort_values(by='Year')
display(df_unpivoted_stock)

df_unpivoted_stock['4_quarter_moving_total'] = df_unpivoted_stock['value'].rolling(window=4).sum()
df_unpivoted_stock['4_quarter_moving_average'] = df_unpivoted_stock['4_quarter_moving_total'] / 4
df_unpivoted_stock['4_quarter_centered_moving_average'] = df_unpivoted_stock['4_quarter_moving_average'].rolling(window=2).mean()
display(df_unpivoted_stock)

df_unpivoted_stock['Percentage_of_Actual_to_Moving_Average'] = (df_unpivoted_stock['value'] / df_unpivoted_stock['4_quarter_centered_moving_average']) * 100
display(df_unpivoted_stock)

modified_seasonal_indices = df_unpivoted_stock.groupby('Quarter')['Percentage_of_Actual_to_Moving_Average'].median()
display(modified_seasonal_indices)

average_modified_seasonal_indices = modified_seasonal_indices.mean()
normalization_factor = 400 / average_modified_seasonal_indices
seasonal_indices = modified_seasonal_indices * normalization_factor
display(seasonal_indices)

seasonal_indices_df = seasonal_indices.reset_index()
seasonal_indices_df.columns = ['Quarter', 'Seasonal_Index']

df_unpivoted_stock = df_unpivoted_stock.merge(seasonal_indices_df, on='Quarter', how='left')

# Handle NaN values in 'Seasonal_Index' by filling with 100 (assuming average seasonality for missing data)
df_unpivoted_stock['Seasonal_Index'].fillna(100, inplace=True)

df_unpivoted_stock['Deseasonalized_Value'] = df_unpivoted_stock['value'] / (df_unpivoted_stock['Seasonal_Index'] / 100)

display(df_unpivoted_stock)

from scipy.stats import linregress

df_deseasonalized = df_unpivoted_stock.dropna(subset=['Deseasonalized_Value']).copy()

# Convert 'Year' to numeric
df_deseasonalized['Year'] = pd.to_numeric(df_deseasonalized['Year'])

slope, intercept, r_value, p_value, std_err = linregress(df_deseasonalized['Year'], df_deseasonalized['Deseasonalized_Value'])

print(f"Slope: {slope}")
print(f"Intercept: {intercept}")

df_deseasonalized['Predicted_Deseasonalized_Value'] = slope * df_deseasonalized['Year'] + intercept
df_deseasonalized['Relative_Cyclical_Residual'] = ((df_deseasonalized['Deseasonalized_Value'] - df_deseasonalized['Predicted_Deseasonalized_Value']) / df_deseasonalized['Predicted_Deseasonalized_Value']) * 100
display(df_deseasonalized)

plt.figure(figsize=(12, 7))
plt.plot(df_unpivoted_stock.index, df_unpivoted_stock['value'], label='Original Data')
plt.plot(df_deseasonalized.index, df_deseasonalized['Deseasonalized_Value'], label='Deseasonalized Data')
plt.plot(df_deseasonalized.index, df_deseasonalized['Predicted_Deseasonalized_Value'], label='Trend Line')
plt.xlabel('Time Period')
plt.ylabel('Value')
plt.title('Original Data, Deseasonalized Data, and Trend Line')
plt.legend()
plt.show()

path = '/content/drive/MyDrive/sem 8/Computational Finance/'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.stats as stats

def regression_model(traindata, testdata):
  traindata['Date'] = pd.to_datetime(traindata['Date'])
  testdata['Date'] = pd.to_datetime(testdata['Date'])
  min_date = traindata['Date'].min()
  traindata['Days'] = (traindata['Date'] - min_date).dt.days
  testdata['Days'] = (testdata['Date'] - min_date).dt.days
  traindata['Rolling_Mean_7'] = traindata['Close'].rolling(window=7).mean().shift(1)
  traindata['Rolling_Std_7'] = traindata['Close'].rolling(window=7).std().shift(1)
  traindata['Prev_Close'] = traindata['Close'].shift(1)
  traindata = traindata.dropna()
  features = ['Days', 'Rolling_Mean_7', 'Rolling_Std_7', 'Prev_Close']
  X_train = traindata[features]
  y_train = traindata['Close']
  testdata['Rolling_Mean_7'] = traindata['Close'].rolling(window=7).mean().iloc[-1]
  testdata['Rolling_Std_7'] = traindata['Close'].rolling(window=7).std().iloc[-1]
  testdata['Prev_Close'] = traindata['Close'].iloc[-1]
  X_test = testdata[features]
  y_test = testdata['Close']
  models = [
  LinearRegression(),
  make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=1.0)),
  make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=0.1))
  ]
  best_model = None
  best_score = float('inf')

  for model in models:
    cv_scores = cross_val_score(
    model,
    X_train,
    y_train,
    scoring='neg_mean_squared_error',
    cv=5
    )
    mean_mse = -cv_scores.mean()
    if mean_mse < best_score:
      best_score = mean_mse
      best_model = model
  best_model.fit(X_train, y_train)
  y_pred = best_model.predict(X_test)
  mse = mean_squared_error(y_test, y_pred)
  mae = mean_absolute_error(y_test, y_pred)
  r2 = r2_score(y_test, y_pred)
  ttest=stats.ttest_ind(y_test, y_pred)
  plt.figure(figsize=(12, 6))
  plt.scatter(X_test['Days'], y_test, label='Actual', color='blue',marker='o')
  plt.scatter(X_test['Days'], y_pred, label='Predicted', color='red',marker='x')
  plt.title('Regression: Actual vs Predicted Prices')
  plt.xlabel('Days')
  plt.ylabel('Close Price')
  plt.legend()
  plt.tight_layout()
  plt.show()
  print(f"Mean Squared Error: {mse}")
  print(f"Mean Absolute Error: {mae}")
  print(f"R-squared Score: {r2}")
  print(f"T-test:{ttest}")
  comparison = pd.DataFrame({
  'Actual': y_test,
  'Predicted': y_pred,
  'Difference': y_test - y_pred
  })
  15
  print(comparison)
  return best_model, y_pred, comparison
traindata=pd.read_csv(path+'16to24.csv')
testdata=pd.read_csv(path+'test.csv')
regression_model(traindata, testdata)

