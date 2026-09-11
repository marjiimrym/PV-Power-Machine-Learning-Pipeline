import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#SECTION 1: Loading Data:
#This sections ensures that solar_data.csv is in the same folder as this code

df = pd.read_csv("solar_data.csv", parse_dates =["DateTime"])
df = df.dropna().sort_values("DateTime").reset_index(drop=True)
print(f"Loading {len(df):,} rows | {df['DateTime'].min().date()} | {df['DateTime'].max().date()}")

#SECTION 2: Feature Engineering
df["Hour"] = df ["DateTime"].dt.hour
df["Month"] = df["DateTime"].dt.month
df["DayOfYear"] = df["DateTime"].dt.dayofyear
df["IsDaytime"] = (df["GHI"] >0).astype(int)
df["Year"] = df["DateTime"].dt.year

Features = ["Temperature", "GHI", "Hour", "DayOfYear", "IsDaytime"]
Target = "PV_Power"

#SECTION 3: Training & Testing Split (in chronological order)
train = df[df["Year"] <= 2018]
test = df[df["Year"] == 2019]

X_train, y_train = train[Features].values, train[Target].values
X_test, y_test = test[Features].values, test[Target].values
print(f"Train: {len(train):,} rows | Test: {len(test):,} rows")

#scale features
scaler = StandardScaler()
Xs_train = scaler.fit_transform(X_train)
Xs_test = scaler.transform(X_test)

#SECTION 4: Train Models
linear_regress = LinearRegression()
linear_regress.fit(Xs_train, y_train)
pred_linear_regress = linear_regress.predict(Xs_test)

random_forest = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs= -1)
random_forest.fit(X_train, y_train)
pred_random_forest = random_forest.predict(X_test)

#SECTION 5: Evaluation
def metrics(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print(f"MAE = {mae: .4f} kW")
    print(f"RMSE = {rmse:.4f} kW")
    print(f"R² = {r2:.4f}")
    return mae, rmse, r2
print("\n -Model Evaluation- ")
metrics(y_test,pred_linear_regress, "Linear Regression")
metrics(y_test, pred_random_forest, "Random Forest")

#SECTION 6: Generating a Correlation Headmap
plt.figure(figsize=(8,6))
correlation = df [["Temperature", "GHI", "PV_Power", "Hour", "Month"]].correlation()
sns.heatmap(correlation, annot=True, fmt= ".2f", cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Feature Correlation HeatMap")
plt.tight_layout()
plt.savefig("plot1_heatmap", dpi=150)
plt.close()
print("\n Plot has been saved: plot1_heatmap")

#SECTION 7: Creating an "Actual vs Prediction Plot"
fig, axes = plt.subplots(1,2, figsize=(12,5))
fig.suptitle("Actual vs Predicted PV Power Test Set for 2019")

for ax, name, predictions, in zip(axes, ["Linear Regression", "Random Forest"], [pred_linear_regress, pred_random_forest]):
    ax.scatter(y_test, predictions, alpha=0.1,s=5,color="red")
    lim = [0, max(y_test.max(), predictions.max()) +5]
    ax.plot(lim, lim, "r--", linewidth=1.5, label= "Perfect fit")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Actual kW")
    ax.set_ylabel("Predicted kW")
    ax.set_title(f"{name}\n R²={r2_score(y_test, predictions):.4f}"
                 f"RMSE={np.sqrt(mean_squared_error(y_test, predictions)):.3f} kW")
    ax.legend(fontsize=9)
    ax.grid(True)
plt.tight_layout()
plt.savefig("plot2_actual_vs_predicted.png", dpi = 150)
plt.close()
print("\nPlot has been saved: plot2_actual_vs_predicted.png")

#SECTION 8: Displaying Random Forest Feature Importance
importance = pd.Series(random_forest.feature_importances_, index=Features).sort_values()
plt.figure(figsize=(7,4))
importance.plot(kind="bar", color = "red", edgecolor="white")
plt.xlabel("Importance Score")
plt.title("Random Forest - Feature Importance")
plt.tight_layout()
plt.savefig("plot3_featureimportance.png", dpi=150)
plt.close()
print("\n The plot has been saved: plot3_featureimportance.png")