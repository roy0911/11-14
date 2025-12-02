import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. 載入資料集
data = load_iris()
X = data.data
y = data.target

# 2. 分割數據集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. 訓練 隨機森林 模型
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf.fit(X_train, y_train)

# 4. 保存 隨機森林 模型
joblib.dump(rf, 'random_forest_model.pkl')

# 5. 訓練 XGBoost 模型
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100
)
xgb_model.fit(X_train, y_train)

# 6. 保存 XGBoost 模型
joblib.dump(xgb_model, 'xgboost_model.pkl')

print("模型訓練並保存完成！")
