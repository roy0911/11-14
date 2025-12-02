import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 載入資料
data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 載入模型
rf_model = joblib.load('random_forest_model.pkl')
xgb_model = joblib.load('xgboost_model.pkl')

# 預測
rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

print("隨機森林預測：", rf_pred)
print("XGBoost預測：", xgb_pred)
