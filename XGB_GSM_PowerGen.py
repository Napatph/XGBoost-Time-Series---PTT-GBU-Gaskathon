import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

df = pd.read_csv('GSM_Demand_PowerGen.csv')

df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

df.sort_index(inplace=True)

# Feature Engineering
def create_lag_features(df, col, lags):
    for lag in lags:
        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

def create_moving_avg_features(df, col, windows):
    for window in windows:
        df[f'{col}_ma_{window}'] = df[col].shift(1).rolling(window=window).mean()
    return df

def create_rolling_std_features(df, col, windows):
    for window in windows:
        df[f'{col}_std_{window}'] = df[col].shift(1).rolling(window=window).std()
    return df

def create_ema_features(df, col, spans):
    for span in spans:
        df[f'{col}_ema_{span}'] = df[col].shift(1).ewm(span=span, adjust=False).mean()
    return df

lags = [1, 2, 3, 5, 7]
windows = [7]
ema_spans = [7]

df = create_lag_features(df, 'Price', lags)
df = create_moving_avg_features(df, 'Price', windows)
df = create_rolling_std_features(df, 'Price', windows)
df = create_ema_features(df, 'Price', ema_spans)

df['Price_lag_1_ratio_ma_7'] = df['Price_lag_1'] / df['Price_ma_7']
df['Price_diff_7'] = df['Price_lag_1'] - df['Price_lag_7']

df['Price_cumsum'] = df['Price_lag_1'].cumsum()
df['Price_cummean'] = df['Price_cumsum'] / (np.arange(len(df)) + 1)

df['week'] = df.index.isocalendar().week
df['month'] = df.index.month
df['year'] = df.index.year

df.dropna(inplace=True)

X = df.drop(columns=['Price'])
y = df['Price']

train_size = int(len(df) * 0.8)
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

import xgboost as xgb
import numpy as np

def mape_eval(y_pred, dtrain):
    y_true = dtrain.get_label()
    error = np.abs((y_true - y_pred) / y_true)
    return 'mape', float(np.mean(error) * 100)

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    'objective': 'reg:squarederror',
    'learning_rate': 0.001,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'eval_metric': 'rmse',
    'tree_method': 'gpu_hist'
}
def learning_rate_schedule(round_number):
    if round_number < 25000:
        return 0.001
    elif round_number < 27500:
        return 0.0005+
    else:
        return 0.0001

evals_result = {}

evals = [(dtrain, 'train'), (dtest, 'eval')]
model = xgb.train(
    params,
    dtrain,
    num_boost_round=30000,
    evals=evals,
    feval=mape_eval,
    verbose_eval=10,
    evals_result=evals_result,
    callbacks=[LearningRateScheduler(learning_rate_schedule)]
)

metrics_df = pd.DataFrame({
    'train-rmse': evals_result['train']['rmse'],
    'eval-rmse': evals_result['eval']['rmse']
})
metrics_df.to_csv("GSM_PowerGen_XGB_v1.csv", index=False)

train_mape = []
eval_mape = []

for i in range(len(evals_result['train']['rmse'])):
    train_pred = model.predict(dtrain, iteration_range=(0, i + 1))
    eval_pred = model.predict(dtest, iteration_range=(0, i + 1))

    train_mape.append(np.mean(np.abs((y_train - train_pred) / y_train)) * 100)
    eval_mape.append(np.mean(np.abs((y_test - eval_pred) / y_test)) * 100)

metrics_df['train-mape'] = train_mape
metrics_df['eval-mape'] = eval_mape
metrics_df.to_csv("GSM_PowerGen_XGB_v1_MAPE.csv", index=False)
y_pred = model.predict(dtest)

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"Test MAE: {mae:.2f}")
print(f"Test RMSE: {rmse:.2f}")
print(f"Test MAPE: {mape:.2f}%")
model.save_model("XGB_GSM_PowerGen_Model_v1.json")


from xgboost import XGBRegressor
loaded_model = XGBRegressor()
loaded_model.load_model("XGB_GSM_PowerGen_Model_v1.json")

y_pred = loaded_model.predict(X_test)


# Evaluate the model
print('Load the saved model')
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE on test set: {rmse:.4f}')

# MAPE calculation
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print(f'MAPE on test set: {mape:.4f}%')

X_test.to_csv('XGB_GSM_PowerGen_X_test.csv', index=True)

y_test.to_csv('XGB_GSM_PowerGen_y_test.csv', index=True, header=True)