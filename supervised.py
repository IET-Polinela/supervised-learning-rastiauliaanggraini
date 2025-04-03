import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Tentukan folder visualizations dan path absolut
visualization_dir = os.path.join(os.getcwd(), 'visualizations')

# Pastikan folder visualizations ada, jika tidak, buat folder tersebut
if not os.path.exists(visualization_dir):
    try:
        os.makedirs(visualization_dir)
        print(f"Folder '{visualization_dir}' berhasil dibuat.")
    except Exception as e:
        print(f"Error membuat folder '{visualization_dir}': {e}")
else:
    print(f"Folder '{visualization_dir}' sudah ada.")

# Load dataset
file_path = "train.csv"  # Sesuaikan dengan lokasi file

df = pd.read_csv(file_path)

# 1. Data Understanding
numeric_cols = df.select_dtypes(include=[np.number]).columns  # Hanya kolom numerik
numeric_stats = df[numeric_cols].describe().T
numeric_stats["median"] = df[numeric_cols].median()
numeric_stats = numeric_stats[["count", "mean", "median", "std", "min", "25%", "50%", "75%", "max"]]
print(numeric_stats)

# Save Data Understanding visualization
plt.figure(figsize=(10, 6))

# Menggunakan hanya kolom numerik untuk korelasi
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(visualization_dir, "data_understanding_heatmap.png"))
plt.close()

# 2. Data Preprocessing
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))  # Handle NaN values
    label_encoders[col] = le

X = df.drop(columns=['SalePrice'])
y = df['SalePrice']

# Menggunakan SimpleImputer untuk menangani nilai NaN
imputer = SimpleImputer(strategy='median')
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)  # Pastikan X tetap menjadi DataFrame

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Outlier Handling
# Boxplot untuk semua fitur numerik
plt.figure(figsize=(15, 8))
X.select_dtypes(include=['number']).boxplot(rot=90, grid=False)
plt.title("Boxplot dari Semua Fitur Numerik")
plt.xticks(rotation=90)
plt.savefig(os.path.join(visualization_dir, "boxplot_fitur_numerik.png"))  # Menyimpan visualisasi boxplot
plt.close()

# Metode IQR untuk menangani outlier
Q1 = X_train.quantile(0.25)  # Pastikan X_train tetap DataFrame
Q3 = X_train.quantile(0.75)
IQR = Q3 - Q1
X_train_no_outliers = X_train[~((X_train < (Q1 - 1.5 * IQR)) | (X_train > (Q3 + 1.5 * IQR))).any(axis=1)]
y_train_no_outliers = y_train.loc[X_train_no_outliers.index]

# 4. Feature Scaling
scalers = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler()}
scaled_data = {}
for name, scaler in scalers.items():
    scaler.fit(X_train_no_outliers)
    scaled_data[name] = scaler.transform(X_train_no_outliers)
    plt.hist(scaled_data[name], bins=50, alpha=0.5, label=name)
plt.legend()
plt.title("Distribusi Data Sebelum dan Sesudah Scaling")
plt.savefig(os.path.join(visualization_dir, "distribusi_data_scaling.png"))  # Menyimpan distribusi data
plt.close()

# 5. Implementation: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# Save Linear Regression results visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Linear Regression: Predicted vs Actual")
plt.savefig(os.path.join(visualization_dir, "linear_regression_results.png"))
plt.close()

# 6. Implementation: Polynomial Regression
poly_degrees = [2, 3]
results_poly = {}
for d in poly_degrees:
    poly = PolynomialFeatures(degree=d)
    X_poly_train = poly.fit_transform(X_train_no_outliers)
    X_poly_test = poly.transform(X_test)
    lr_poly = LinearRegression()
    lr_poly.fit(X_poly_train, y_train_no_outliers)
    y_pred_poly = lr_poly.predict(X_poly_test)
    mse_poly = mean_squared_error(y_test, y_pred_poly)
    r2_poly = r2_score(y_test, y_pred_poly)
    results_poly[d] = {"MSE": mse_poly, "R2": r2_poly}

# Save Polynomial Regression results visualization
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_poly, alpha=0.5, label="Polynomial Regression")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("Polynomial Regression: Predicted vs Actual")
plt.savefig(os.path.join(visualization_dir, "polynomial_regression_results.png"))
plt.close()

# 7. Implementation: KNN Regression
results_knn = {}
knn_preds = {}  # Menyimpan hasil prediksi untuk setiap K
for k in [3, 5, 7]:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train_no_outliers, y_train_no_outliers)
    y_pred_knn = knn.predict(X_test)
    knn_preds[k] = y_pred_knn  # Simpan hasil prediksi KNN untuk setiap k
    mse_knn = mean_squared_error(y_test, y_pred_knn)
    r2_knn = r2_score(y_test, y_pred_knn)
    results_knn[k] = {"MSE": mse_knn, "R2": r2_knn}

# Save KNN Regression results visualization
plt.figure(figsize=(10, 6))
for k in [3, 5, 7]:
    plt.scatter(y_test, knn_preds[k], alpha=0.5, label=f"KNN Regression (K={k})")
plt.xlabel("Actual SalePrice")
plt.ylabel("Predicted SalePrice")
plt.title("KNN Regression: Predicted vs Actual")
plt.legend()
plt.savefig(os.path.join(visualization_dir, "knn_regression_results.png"))
plt.close()

# 8. Analysis Comparison Models and Conclusion
# Tabel perbandingan MSE dan R2
comparison_df = pd.DataFrame({
    "Model": ["Linear Regression"] + [f"Polynomial Regression (Degree {d})" for d in poly_degrees] + [f"KNN Regression (K={k})" for k in [3, 5, 7]],
    "MSE": [mse_lr] + [results_poly[d]["MSE"] for d in poly_degrees] + [results_knn[k]["MSE"] for k in [3, 5, 7]],
    "R2": [r2_lr] + [results_poly[d]["R2"] for d in poly_degrees] + [results_knn[k]["R2"] for k in [3, 5, 7]]
})

# Save Comparison visualization
plt.figure(figsize=(10, 6))
sns.barplot(x="Model", y="MSE", data=comparison_df)
plt.title("Model Comparison - MSE")
plt.xticks(rotation=45)
plt.savefig(os.path.join(visualization_dir, "model_comparison_mse.png"))
plt.close()
