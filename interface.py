import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict

# Load dataset
df = pd.read_csv("AIDS_Classification.csv")

X = df.drop(columns=['infected'])
y = df['infected']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

skenario_1_results = {'Accuracy': '80.92%', 'Precision': '83.24%', 'Recall': '77.82%', 'F1-Score': '80.32%'}
skenario_2_results = {'Accuracy': '80.70%', 'Precision': '83.13%', 'Recall': '77.21%', 'F1-Score': '79.97%'}
skenario_3_results = {'Accuracy': '80.82%', 'Precision': '82.54%', 'Recall': '78.63%', 'F1-Score': '80.39%'}

# Sidebar navigation
st.sidebar.image("logo.webp", use_container_width=True)
st.sidebar.markdown("<h1 style='text-align: center; color: cyan;'>Klasifikasi HIV</h1>", unsafe_allow_html=True)
st.sidebar.markdown("---") 
st.sidebar.markdown("<p style='text-align: center;'>Analisis menggunakan PCA dan SVM untuk klasifikasi infeksi HIV.</p>", unsafe_allow_html=True)
menu = st.sidebar.radio("📌 Pilih Halaman", 
                        ["📂 Dataset", 
                         "📊 Skenario 1 (90:10)", 
                         "📊 Skenario 2 (80:20)", 
                         "📊 Skenario 3 (70:30)", 
                         "📈 Evaluasi Semua Skenario"])
st.sidebar.markdown("---")
st.sidebar.markdown("<p style='text-align: center; font-size: 12px;'>Developed by Gigih Agung Prasetyo</p>", unsafe_allow_html=True)

if menu == "📂 Dataset":
    st.subheader("Dataset AIDS Classification")
    st.write("Berikut adalah cuplikan dari dataset:")
    st.dataframe(df.head())
    
    st.write("### Statistik Deskriptif")
    st.write(df.describe())
    
    st.write("### Cek Missing Values")
    st.write(df.isnull().sum())
    
    st.write("### Data Setelah Standarisasi")
    st.dataframe(pd.DataFrame(X_scaled, columns=X.columns).head())

elif menu == "📊 Skenario 1 (90:10)":
    st.subheader("Skenario 1: Split Data 90:10")
    test_size = 0.1

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, stratify=y)

    st.write("### Shape Data")
    st.write(f"Training Data: {X_train.shape}, Testing Data: {X_test.shape}")

    train_class_counts = y_train.value_counts().to_frame(name="Jumlah Data Training")
    test_class_counts = y_test.value_counts().to_frame(name="Jumlah Data Testing")

    class_distribution = train_class_counts.join(test_class_counts, how="outer").fillna(0)
    class_distribution.index = ["Non-Infected", "Infected"]
    st.write("### Jumlah Data Training dan Testing berdasarkan Kelas:")
    st.dataframe(class_distribution)

    pca = PCA()
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    st.write("### Eigenvalues")
    st.dataframe(pd.DataFrame({"Eigenvalues": pca.explained_variance_}))

    st.write("### Explained Variance Ratio")
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    df_variance = pd.DataFrame({"Component": range(1, len(explained_variance_ratio) + 1), "Explained Variance": explained_variance_ratio})
    st.line_chart(df_variance.set_index("Component"))

    num_components = np.argmax(explained_variance_ratio >= 0.95) + 1
    st.write(f"Number of components to retain 95% variance: {num_components}")

    pca = PCA(n_components=num_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    st.write("\nPrincipal Components (training data):")
    st.write(pd.DataFrame(X_train_pca).head())

    rus = RandomUnderSampler(random_state=42)
    X_train_balanced, y_train_balanced = rus.fit_resample(X_train_pca[:, :num_components], y_train)
    st.write("### Data Setelah Undersampling")
    st.dataframe(pd.Series(y_train_balanced).value_counts().to_frame(name="Jumlah Data"))

    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10]}
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    grid_search = GridSearchCV(SVC(kernel='rbf', random_state=42), param_grid=param_grid, cv=5, scoring=scoring, refit='accuracy')

    grid_search.fit(X_train_balanced, y_train_balanced)

    results = pd.DataFrame(grid_search.cv_results_)

    if {'mean_test_accuracy', 'mean_test_precision', 'mean_test_recall', 'mean_test_f1'}.issubset(results.columns):
            avg_metrics_by_C = results.groupby('param_C').agg({
                'mean_test_accuracy': 'mean',
                'mean_test_precision': 'mean',
                'mean_test_recall': 'mean',
                'mean_test_f1': 'mean'
            })
            avg_metrics_by_C *= 100
            avg_metrics_by_C = avg_metrics_by_C.round(2).astype(str) + " %"
            avg_metrics_by_C.columns = ['Akurasi', 'Precision', 'Recall', 'F1 Score']
            st.write("### Rata-rata Evaluasi Berdasarkan Nilai C")
            st.dataframe(avg_metrics_by_C)

    if {'mean_test_accuracy', 'mean_test_precision', 'mean_test_recall', 'mean_test_f1'}.issubset(results.columns):
        avg_metrics_by_gamma = results.groupby('param_gamma').agg({
            'mean_test_accuracy': 'mean',
            'mean_test_precision': 'mean',
            'mean_test_recall': 'mean',
            'mean_test_f1': 'mean'
        })
        avg_metrics_by_gamma *= 100
        avg_metrics_by_gamma = avg_metrics_by_gamma.round(2).astype(str) + " %"
        avg_metrics_by_gamma.columns = ['Akurasi', 'Precision', 'Recall', 'F1 Score']
        st.write("### Rata-rata Evaluasi Berdasarkan Nilai Gamma")
        st.dataframe(avg_metrics_by_gamma)

    if 'mean_test_accuracy' in results.columns:
        formatted_results = results[['param_C', 'param_gamma', 
                             'mean_test_accuracy', 'mean_test_precision', 
                             'mean_test_recall', 'mean_test_f1']].copy()
        formatted_results.columns = ['C', 'Gamma', 'Akurasi', 'Precision', 'Recall', 'F1-Score']
        formatted_results[['Akurasi', 'Precision', 'Recall', 'F1-Score']] *= 100  
        st.write("### Hasil Evaluasi Per Kombinasi Hyperparameter")
        st.dataframe(formatted_results)

    avg_accuracy = str(round(results['mean_test_accuracy'].mean() * 100, 2)) + " %"
    avg_precision = str(round(results['mean_test_precision'].mean() * 100, 2)) + " %"
    avg_recall = str(round(results['mean_test_recall'].mean() * 100, 2)) + " %"
    avg_f1 = str(round(results['mean_test_f1'].mean() * 100, 2)) + " %"

    st.write("### Rata-rata Evaluasi Model:")
    evaluation_mean = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [avg_accuracy, avg_precision, avg_recall, avg_f1]
    })
    st.dataframe(evaluation_mean)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    st.write("### Parameter Terbaik Berdasarkan Data Training")
    st.json(best_params)

    y_pred_test = best_model.predict(X_test_pca[:, :num_components])
    accuracy_test = str(round(accuracy_score(y_test, y_pred_test) * 100, 2)) + " %"
    precision_test = str(round(precision_score(y_test, y_pred_test, zero_division=1) * 100, 2)) + " %"
    recall_test = str(round(recall_score(y_test, y_pred_test) * 100, 2)) + " %"
    f1_test = str(round(f1_score(y_test, y_pred_test) * 100, 2)) + " %"

    st.write("### Akurasi Model Terbaik pada Data Testing")
    evaluation_results = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [accuracy_test, precision_test, recall_test, f1_test]
    })
    st.dataframe(evaluation_results)

    best_accuracy = str(round(grid_search.best_score_ * 100, 2)) + " %"
    best_precision = str(round(results['mean_test_precision'][grid_search.best_index_] * 100, 2)) + " %"
    best_recall = str(round(results['mean_test_recall'][grid_search.best_index_] * 100, 2)) + " %"
    best_f1 = str(round(results['mean_test_f1'][grid_search.best_index_] * 100, 2)) + " %"

    st.write("### Hasil Terbaik Dari Skenario 1")
    skenario_1_results = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value (%)": [best_accuracy, best_precision, best_recall, best_f1]
    })
    st.write(skenario_1_results)

    y_pred_cv = cross_val_predict(grid_search.best_estimator_, X_train_balanced, y_train_balanced, cv=5)

    st.write("### Confusion Matrix pada Parameter Terbaik")
    conf_matrix_cv = confusion_matrix(y_train_balanced, y_pred_cv)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(conf_matrix_cv, display_labels=["Non-Infected", "Infected"]).plot(ax=ax)
    st.pyplot(fig)

    st.write("### Confusion Matrix pada Data Testing")
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(conf_matrix, display_labels=["Non-Infected", "Infected"]).plot(ax=ax)
    st.pyplot(fig)

    st.write("### Akurasi pada Setiap Percobaan Tuning Parameter")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(1, len(formatted_results) + 1), formatted_results['Akurasi'], color='blue', alpha=0.7)
    ax.set_xticks(range(1, len(formatted_results) + 1))
    ax.set_xticklabels(range(1, len(formatted_results) + 1))
    ax.set_xlabel("Percobaan Tuning Parameter")
    ax.set_ylabel("Akurasi (%)")
    ax.set_title("Skenario 1: Akurasi pada Setiap Percobaan")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    st.write("### Presisi pada Setiap Percobaan Tuning Parameter")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(1, len(formatted_results) + 1), formatted_results['Precision'], color='blue', alpha=0.7)
    ax.set_xticks(range(1, len(formatted_results) + 1))
    ax.set_xticklabels(range(1, len(formatted_results) + 1))
    ax.set_xlabel("Percobaan Tuning Parameter")
    ax.set_ylabel("Presisi (%)")
    ax.set_title("Skenario 1: Presisi pada Setiap Percobaan")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    st.write("### Recall pada Setiap Percobaan Tuning Parameter")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(1, len(formatted_results) + 1), formatted_results['Recall'], color='blue', alpha=0.7)
    ax.set_xticks(range(1, len(formatted_results) + 1))
    ax.set_xticklabels(range(1, len(formatted_results) + 1))
    ax.set_xlabel("Percobaan Tuning Parameter")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Skenario 1: Recall pada Setiap Percobaan")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    st.write("### F1-Score pada Setiap Percobaan Tuning Parameter")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(1, len(formatted_results) + 1), formatted_results['F1-Score'], color='blue', alpha=0.7)
    ax.set_xticks(range(1, len(formatted_results) + 1))
    ax.set_xticklabels(range(1, len(formatted_results) + 1))
    ax.set_xlabel("Percobaan Tuning Parameter")
    ax.set_ylabel("F1-Score (%)")
    ax.set_title("Skenario 1: F1-Score pada Setiap Percobaan")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

elif menu == "📊 Skenario 2 (80:20)":
    st.subheader("Skenario 2: Split Data 80:20")
    test_size = 0.2

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, stratify=y)

    st.write("### Shape Data")
    st.write(f"Training Data: {X_train.shape}, Testing Data: {X_test.shape}")

    train_class_counts = y_train.value_counts().to_frame(name="Jumlah Data Training")
    test_class_counts = y_test.value_counts().to_frame(name="Jumlah Data Testing")

    class_distribution = train_class_counts.join(test_class_counts, how="outer").fillna(0)
    class_distribution.index = ["Non-Infected", "Infected"]
    st.write("### Jumlah Data Training dan Testing berdasarkan Kelas:")
    st.dataframe(class_distribution)

    pca = PCA()
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    st.write("### Eigenvalues")
    st.dataframe(pd.DataFrame({"Eigenvalues": pca.explained_variance_}))

    st.write("### Explained Variance Ratio")
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    df_variance = pd.DataFrame({"Component": range(1, len(explained_variance_ratio) + 1), "Explained Variance": explained_variance_ratio})
    st.line_chart(df_variance.set_index("Component"))

    num_components = np.argmax(explained_variance_ratio >= 0.95) + 1
    st.write(f"Number of components to retain 95% variance: {num_components}")

    pca = PCA(n_components=num_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    st.write("\nPrincipal Components (training data):")
    st.write(pd.DataFrame(X_train_pca).head())

    rus = RandomUnderSampler(random_state=42)
    X_train_balanced, y_train_balanced = rus.fit_resample(X_train_pca[:, :num_components], y_train)
    st.write("### Data Setelah Undersampling")
    st.dataframe(pd.Series(y_train_balanced).value_counts().to_frame(name="Jumlah Data"))

    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10]}
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    grid_search = GridSearchCV(SVC(kernel='rbf', random_state=42), param_grid=param_grid, cv=5, scoring=scoring, refit='accuracy')

    grid_search.fit(X_train_balanced, y_train_balanced)

    results = pd.DataFrame(grid_search.cv_results_)

    if {'mean_test_accuracy', 'mean_test_precision', 'mean_test_recall', 'mean_test_f1'}.issubset(results.columns):
            avg_metrics_by_C = results.groupby('param_C').agg({
                'mean_test_accuracy': 'mean',
                'mean_test_precision': 'mean',
                'mean_test_recall': 'mean',
                'mean_test_f1': 'mean'
            })
            avg_metrics_by_C *= 100
            avg_metrics_by_C = avg_metrics_by_C.round(2).astype(str) + " %"
            avg_metrics_by_C.columns = ['Akurasi', 'Precision', 'Recall', 'F1 Score']
            st.write("### Rata-rata Evaluasi Berdasarkan Nilai C")
            st.dataframe(avg_metrics_by_C)

    if {'mean_test_accuracy', 'mean_test_precision', 'mean_test_recall', 'mean_test_f1'}.issubset(results.columns):
        avg_metrics_by_gamma = results.groupby('param_gamma').agg({
            'mean_test_accuracy': 'mean',
            'mean_test_precision': 'mean',
            'mean_test_recall': 'mean',
            'mean_test_f1': 'mean'
        })
        avg_metrics_by_gamma *= 100
        avg_metrics_by_gamma = avg_metrics_by_gamma.round(2).astype(str) + " %"
        avg_metrics_by_gamma.columns = ['Akurasi', 'Precision', 'Recall', 'F1 Score']
        st.write("### Rata-rata Evaluasi Berdasarkan Nilai Gamma")
        st.dataframe(avg_metrics_by_gamma)

    if 'mean_test_accuracy' in results.columns:
        formatted_results = results[['param_C', 'param_gamma', 
                             'mean_test_accuracy', 'mean_test_precision', 
                             'mean_test_recall', 'mean_test_f1']].copy()
        formatted_results.columns = ['C', 'Gamma', 'Akurasi', 'Precision', 'Recall', 'F1-Score']
        formatted_results[['Akurasi', 'Precision', 'Recall', 'F1-Score']] *= 100  
        st.write("### Hasil Evaluasi Per Kombinasi Hyperparameter")
        st.dataframe(formatted_results)

    avg_accuracy = str(round(results['mean_test_accuracy'].mean() * 100, 2)) + " %"
    avg_precision = str(round(results['mean_test_precision'].mean() * 100, 2)) + " %"
    avg_recall = str(round(results['mean_test_recall'].mean() * 100, 2)) + " %"
    avg_f1 = str(round(results['mean_test_f1'].mean() * 100, 2)) + " %"

    st.write("### Rata-rata Evaluasi Model:")
    evaluation_mean = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [avg_accuracy, avg_precision, avg_recall, avg_f1]
    })
    st.dataframe(evaluation_mean)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    st.write("### Parameter Terbaik Berdasarkan Data Training")
    st.json(best_params)

    y_pred_test = best_model.predict(X_test_pca[:, :num_components])
    accuracy_test = str(round(accuracy_score(y_test, y_pred_test) * 100, 2)) + " %"
    precision_test = str(round(precision_score(y_test, y_pred_test, zero_division=1) * 100, 2)) + " %"
    recall_test = str(round(recall_score(y_test, y_pred_test) * 100, 2)) + " %"
    f1_test = str(round(f1_score(y_test, y_pred_test) * 100, 2)) + " %"

    st.write("### Akurasi Model Terbaik pada Data Testing")
    evaluation_results = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [accuracy_test, precision_test, recall_test, f1_test]
    })
    st.dataframe(evaluation_results)

    best_accuracy = str(round(grid_search.best_score_ * 100, 2)) + " %"
    best_precision = str(round(results['mean_test_precision'][grid_search.best_index_] * 100, 2)) + " %"
    best_recall = str(round(results['mean_test_recall'][grid_search.best_index_] * 100, 2)) + " %"
    best_f1 = str(round(results['mean_test_f1'][grid_search.best_index_] * 100, 2)) + " %"

    st.write("### Hasil Terbaik Dari Skenario 2")
    skenario_2_results = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value (%)": [best_accuracy, best_precision, best_recall, best_f1]
    })
    st.write(skenario_2_results)

    y_pred_cv = cross_val_predict(grid_search.best_estimator_, X_train_balanced, y_train_balanced, cv=5)

    st.write("### Confusion Matrix pada Parameter Terbaik")
    conf_matrix_cv = confusion_matrix(y_train_balanced, y_pred_cv)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(conf_matrix_cv, display_labels=["Non-Infected", "Infected"]).plot(ax=ax)
    st.pyplot(fig)

    st.write("### Confusion Matrix pada Data Testing")
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(conf_matrix, display_labels=["Non-Infected", "Infected"]).plot(ax=ax)
    st.pyplot(fig)

    st.write("### Akurasi pada Setiap Percobaan Tuning Parameter")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(1, len(formatted_results) + 1), formatted_results['Akurasi'], color='blue', alpha=0.7)
    ax.set_xticks(range(1, len(formatted_results) + 1))
    ax.set_xticklabels(range(1, len(formatted_results) + 1))
    ax.set_xlabel("Percobaan Tuning Parameter")
    ax.set_ylabel("Akurasi (%)")
    ax.set_title("Skenario 2: Akurasi pada Setiap Percobaan")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    st.write("### Presisi pada Setiap Percobaan Tuning Parameter")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(1, len(formatted_results) + 1), formatted_results['Precision'], color='blue', alpha=0.7)
    ax.set_xticks(range(1, len(formatted_results) + 1))
    ax.set_xticklabels(range(1, len(formatted_results) + 1))
    ax.set_xlabel("Percobaan Tuning Parameter")
    ax.set_ylabel("Presisi (%)")
    ax.set_title("Skenario 2: Presisi pada Setiap Percobaan")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    st.write("### Recall pada Setiap Percobaan Tuning Parameter")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(1, len(formatted_results) + 1), formatted_results['Recall'], color='blue', alpha=0.7)
    ax.set_xticks(range(1, len(formatted_results) + 1))
    ax.set_xticklabels(range(1, len(formatted_results) + 1))
    ax.set_xlabel("Percobaan Tuning Parameter")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Skenario 2: Recall pada Setiap Percobaan")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    st.write("### F1-Score pada Setiap Percobaan Tuning Parameter")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(1, len(formatted_results) + 1), formatted_results['F1-Score'], color='blue', alpha=0.7)
    ax.set_xticks(range(1, len(formatted_results) + 1))
    ax.set_xticklabels(range(1, len(formatted_results) + 1))
    ax.set_xlabel("Percobaan Tuning Parameter")
    ax.set_ylabel("F1-Score (%)")
    ax.set_title("Skenario 2: F1-Score pada Setiap Percobaan")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

elif menu == "📊 Skenario 3 (70:30)":
    st.subheader("Skenario 3: Split Data 70:30")
    test_size = 0.3

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, stratify=y)

    st.write("### Shape Data")
    st.write(f"Training Data: {X_train.shape}, Testing Data: {X_test.shape}")

    train_class_counts = y_train.value_counts().to_frame(name="Jumlah Data Training")
    test_class_counts = y_test.value_counts().to_frame(name="Jumlah Data Testing")

    class_distribution = train_class_counts.join(test_class_counts, how="outer").fillna(0)
    class_distribution.index = ["Non-Infected", "Infected"]
    st.write("### Jumlah Data Training dan Testing berdasarkan Kelas:")
    st.dataframe(class_distribution)

    pca = PCA()
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    st.write("### Eigenvalues")
    st.dataframe(pd.DataFrame({"Eigenvalues": pca.explained_variance_}))

    st.write("### Explained Variance Ratio")
    explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
    df_variance = pd.DataFrame({"Component": range(1, len(explained_variance_ratio) + 1), "Explained Variance": explained_variance_ratio})
    st.line_chart(df_variance.set_index("Component"))

    num_components = np.argmax(explained_variance_ratio >= 0.95) + 1
    st.write(f"Number of components to retain 95% variance: {num_components}")

    pca = PCA(n_components=num_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    st.write("\nPrincipal Components (training data):")
    st.write(pd.DataFrame(X_train_pca).head())

    rus = RandomUnderSampler(random_state=42)
    X_train_balanced, y_train_balanced = rus.fit_resample(X_train_pca[:, :num_components], y_train)
    st.write("### Data Setelah Undersampling")
    st.dataframe(pd.Series(y_train_balanced).value_counts().to_frame(name="Jumlah Data"))

    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.1, 1, 10]}
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    grid_search = GridSearchCV(SVC(kernel='rbf', random_state=42), param_grid=param_grid, cv=5, scoring=scoring, refit='accuracy')

    grid_search.fit(X_train_balanced, y_train_balanced)

    results = pd.DataFrame(grid_search.cv_results_)

    if {'mean_test_accuracy', 'mean_test_precision', 'mean_test_recall', 'mean_test_f1'}.issubset(results.columns):
            avg_metrics_by_C = results.groupby('param_C').agg({
                'mean_test_accuracy': 'mean',
                'mean_test_precision': 'mean',
                'mean_test_recall': 'mean',
                'mean_test_f1': 'mean'
            })
            avg_metrics_by_C *= 100
            avg_metrics_by_C = avg_metrics_by_C.round(2).astype(str) + " %"
            avg_metrics_by_C.columns = ['Akurasi', 'Precision', 'Recall', 'F1 Score']
            st.write("### Rata-rata Evaluasi Berdasarkan Nilai C")
            st.dataframe(avg_metrics_by_C)

    if {'mean_test_accuracy', 'mean_test_precision', 'mean_test_recall', 'mean_test_f1'}.issubset(results.columns):
        avg_metrics_by_gamma = results.groupby('param_gamma').agg({
            'mean_test_accuracy': 'mean',
            'mean_test_precision': 'mean',
            'mean_test_recall': 'mean',
            'mean_test_f1': 'mean'
        })
        avg_metrics_by_gamma *= 100
        avg_metrics_by_gamma = avg_metrics_by_gamma.round(2).astype(str) + " %"
        avg_metrics_by_gamma.columns = ['Akurasi', 'Precision', 'Recall', 'F1 Score']
        st.write("### Rata-rata Evaluasi Berdasarkan Nilai Gamma")
        st.dataframe(avg_metrics_by_gamma)

    if 'mean_test_accuracy' in results.columns:
        formatted_results = results[['param_C', 'param_gamma', 
                             'mean_test_accuracy', 'mean_test_precision', 
                             'mean_test_recall', 'mean_test_f1']].copy()
        formatted_results.columns = ['C', 'Gamma', 'Akurasi', 'Precision', 'Recall', 'F1-Score']
        formatted_results[['Akurasi', 'Precision', 'Recall', 'F1-Score']] *= 100  
        st.write("### Hasil Evaluasi Per Kombinasi Hyperparameter")
        st.dataframe(formatted_results)

    avg_accuracy = str(round(results['mean_test_accuracy'].mean() * 100, 2)) + " %"
    avg_precision = str(round(results['mean_test_precision'].mean() * 100, 2)) + " %"
    avg_recall = str(round(results['mean_test_recall'].mean() * 100, 2)) + " %"
    avg_f1 = str(round(results['mean_test_f1'].mean() * 100, 2)) + " %"

    st.write("### Rata-rata Evaluasi Model:")
    evaluation_mean = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [avg_accuracy, avg_precision, avg_recall, avg_f1]
    })
    st.dataframe(evaluation_mean)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    st.write("### Parameter Terbaik Berdasarkan Data Training")
    st.json(best_params)

    y_pred_test = best_model.predict(X_test_pca[:, :num_components])
    accuracy_test = str(round(accuracy_score(y_test, y_pred_test) * 100, 2)) + " %"
    precision_test = str(round(precision_score(y_test, y_pred_test, zero_division=1) * 100, 2)) + " %"
    recall_test = str(round(recall_score(y_test, y_pred_test) * 100, 2)) + " %"
    f1_test = str(round(f1_score(y_test, y_pred_test) * 100, 2)) + " %"

    st.write("### Akurasi Model Terbaik pada Data Testing")
    evaluation_results = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [accuracy_test, precision_test, recall_test, f1_test]
    })
    st.dataframe(evaluation_results)

    best_accuracy = str(round(grid_search.best_score_ * 100, 2)) + " %"
    best_precision = str(round(results['mean_test_precision'][grid_search.best_index_] * 100, 2)) + " %"
    best_recall = str(round(results['mean_test_recall'][grid_search.best_index_] * 100, 2)) + " %"
    best_f1 = str(round(results['mean_test_f1'][grid_search.best_index_] * 100, 2)) + " %"

    st.write("### Hasil Terbaik Dari Skenario 3")
    skenario_3_results = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value (%)": [best_accuracy, best_precision, best_recall, best_f1]
    })
    st.write(skenario_3_results)

    y_pred_cv = cross_val_predict(grid_search.best_estimator_, X_train_balanced, y_train_balanced, cv=5)

    st.write("### Confusion Matrix pada Parameter Terbaik")
    conf_matrix_cv = confusion_matrix(y_train_balanced, y_pred_cv)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(conf_matrix_cv, display_labels=["Non-Infected", "Infected"]).plot(ax=ax)
    st.pyplot(fig)

    st.write("### Confusion Matrix pada Data Testing")
    conf_matrix = confusion_matrix(y_test, y_pred_test)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(conf_matrix, display_labels=["Non-Infected", "Infected"]).plot(ax=ax)
    st.pyplot(fig)

    st.write("### Akurasi pada Setiap Percobaan Tuning Parameter")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(1, len(formatted_results) + 1), formatted_results['Akurasi'], color='blue', alpha=0.7)
    ax.set_xticks(range(1, len(formatted_results) + 1))
    ax.set_xticklabels(range(1, len(formatted_results) + 1))
    ax.set_xlabel("Percobaan Tuning Parameter")
    ax.set_ylabel("Akurasi (%)")
    ax.set_title("Skenario 3: Akurasi pada Setiap Percobaan")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    st.write("### Presisi pada Setiap Percobaan Tuning Parameter")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(1, len(formatted_results) + 1), formatted_results['Precision'], color='blue', alpha=0.7)
    ax.set_xticks(range(1, len(formatted_results) + 1))
    ax.set_xticklabels(range(1, len(formatted_results) + 1))
    ax.set_xlabel("Percobaan Tuning Parameter")
    ax.set_ylabel("Presisi (%)")
    ax.set_title("Skenario 3: Presisi pada Setiap Percobaan")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    st.write("### Recall pada Setiap Percobaan Tuning Parameter")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(1, len(formatted_results) + 1), formatted_results['Recall'], color='blue', alpha=0.7)
    ax.set_xticks(range(1, len(formatted_results) + 1))
    ax.set_xticklabels(range(1, len(formatted_results) + 1))
    ax.set_xlabel("Percobaan Tuning Parameter")
    ax.set_ylabel("Recall (%)")
    ax.set_title("Skenario 3: Recall pada Setiap Percobaan")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

    st.write("### F1-Score pada Setiap Percobaan Tuning Parameter")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(range(1, len(formatted_results) + 1), formatted_results['F1-Score'], color='blue', alpha=0.7)
    ax.set_xticks(range(1, len(formatted_results) + 1))
    ax.set_xticklabels(range(1, len(formatted_results) + 1))
    ax.set_xlabel("Percobaan Tuning Parameter")
    ax.set_ylabel("F1-Score (%)")
    ax.set_title("Skenario 3: F1-Score pada Setiap Percobaan")
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    st.pyplot(fig)

elif menu == "📈 Evaluasi Semua Skenario":
    st.subheader("Evaluasi Semua Skenario")
    
    # Membuat DataFrame hasil evaluasi
    results_df = pd.DataFrame([
        {'Skenario': '90:10', **skenario_1_results},
        {'Skenario': '80:20', **skenario_2_results},
        {'Skenario': '70:30', **skenario_3_results}
    ])

    # 🔥 Konversi nilai persentase menjadi float
    for col in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        results_df[col] = results_df[col].astype(str).str.rstrip('%')  # Hapus simbol %
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')  # Konversi ke float

    # Menampilkan tabel dengan format angka 2 desimal
    st.write("📊 **Hasil Evaluasi Semua Skenario**")
    st.dataframe(results_df.style.format({
        'Accuracy': "{:.2f}%",
        'Precision': "{:.2f}%",
        'Recall': "{:.2f}%",
        'F1-Score': "{:.2f}%"
    }))

    # Plot perbandingan akurasi
    st.write("📈 **Grafik Perbandingan Akurasi**")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(results_df['Skenario'], results_df['Accuracy'], color=['blue', 'green', 'red'], alpha=0.7)
    ax.set_title("Perbandingan Akurasi untuk Setiap Skenario")
    ax.set_xlabel("Skenario (Split Data)")
    ax.set_ylabel("Akurasi (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Tampilkan plot di Streamlit
    st.pyplot(fig)