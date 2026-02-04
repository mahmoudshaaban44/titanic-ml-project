import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='Titanic Flexible Analysis', layout='centered')
st.title("Titanic Dataset Flexible Project")

# ===================== Load Dataset =====================
st.header("Load Dataset")
train_file = st.file_uploader("Upload Titanic training CSV file", type=["csv"])
test_file = st.file_uploader("Upload Titanic test CSV file", type=["csv"])

train_df = None
test_df = None

if train_file and test_file:
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    st.success("Dataset loaded successfully!")

if st.button("SHOW DATA") and train_df is not None:
    st.subheader("Training Data")
    st.write(train_df.head())
    st.subheader("Test Data")
    st.write(test_df.head())

# ===================== Main Pipeline =====================
if train_df is not None and test_df is not None:

    # ===================== Data Cleaning =====================
    st.header("Data Cleaning")
    if st.button("Clean Data"):
        try:
            if 'Age' in train_df.columns:
                train_df['Age'].fillna(train_df['Age'].median(), inplace=True)

            if 'Embarked' in train_df.columns:
                train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace=True)

            if 'Cabin' in train_df.columns:
                train_df.drop('Cabin', axis=1, inplace=True)
                test_df.drop('Cabin', axis=1, inplace=True)

            numeric_cols = train_df.select_dtypes(include=['int64','float64']).columns
            for col in numeric_cols:
                Q1 = train_df[col].quantile(0.25)
                Q3 = train_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                train_df = train_df[(train_df[col] >= lower) & (train_df[col] <= upper)]

            st.success("Data cleaning completed!")
            st.write(train_df.head())

        except Exception as e:
            st.error(f"Cleaning error: {e}")

    # ===================== Data Visualization =====================
    st.header("Data Visualization")

    numeric_cols = train_df.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = train_df.select_dtypes(include=['object']).columns.tolist()

    plot_type = st.radio("Select Plot Type", ["Scatter Plot", "Bar Plot", "Box Plot"])

    if plot_type == "Scatter Plot":
        x_col = st.selectbox("X (numeric)", numeric_cols)
        y_col = st.selectbox("Y (numeric)", numeric_cols)
        hue_col = st.selectbox("Color by (optional)", ["None"] + cat_cols)

    elif plot_type == "Bar Plot":
        x_col = st.selectbox("X (category)", cat_cols)
        y_col = st.selectbox("Y (numeric)", numeric_cols)
        hue_col = st.selectbox("Color by (optional)", ["None"] + cat_cols)

    elif plot_type == "Box Plot":
        x_col = st.selectbox("X (category)", cat_cols)
        y_col = st.selectbox("Y (numeric)", numeric_cols)
        hue_col = st.selectbox("Color by (optional)", ["None"] + cat_cols)

    if st.button("Generate Plot"):
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots(figsize=(8,5))
        hue_val = None if hue_col == "None" else hue_col

        if plot_type == "Scatter Plot":
            sns.scatterplot(train_df, x=x_col, y=y_col, hue=hue_val, palette="viridis", s=70, ax=ax)
        elif plot_type == "Bar Plot":
            sns.barplot(train_df, x=x_col, y=y_col, hue=hue_val, palette="coolwarm", ax=ax)
        elif plot_type == "Box Plot":
            sns.boxplot(train_df, x=x_col, y=y_col, hue=hue_val, palette="Set2", ax=ax)

        ax.set_title(f"{plot_type}: {x_col} vs {y_col}")
        st.pyplot(fig)

    # ===================== Preprocessing =====================
    st.header("Preprocessing")

    target = st.selectbox("Target Column", train_df.columns,
                          index=train_df.columns.get_loc('Survived'))

    feature_cols = st.multiselect("Feature Columns",
                                  [c for c in train_df.columns if c != target])

    if st.button("Implement Preprocessing"):
        try:
            for col in feature_cols:
                if train_df[col].dtype == 'object':
                    le = LabelEncoder()
                    train_df[col] = le.fit_transform(train_df[col].astype(str))
                    test_df[col] = le.transform(test_df[col].astype(str))

            imputer = SimpleImputer(strategy='mean')
            scaler = StandardScaler()

            train_df[feature_cols] = imputer.fit_transform(train_df[feature_cols])
            test_df[feature_cols] = imputer.transform(test_df[feature_cols])

            train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
            test_df[feature_cols] = scaler.transform(test_df[feature_cols])

            X = train_df[feature_cols]
            y = train_df[target]

            # === REAL EVALUATION SPLIT ===
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            st.session_state['X_train'] = X_train
            st.session_state['y_train'] = y_train
            st.session_state['X_val'] = X_val
            st.session_state['y_val'] = y_val
            st.session_state['X_test'] = test_df[feature_cols]
            st.session_state['test_df_original'] = test_df

            st.success("Preprocessing done with Train/Validation split!")
            st.write("Train:", X_train.shape)
            st.write("Validation:", X_val.shape)

        except Exception as e:
            st.error(f"Preprocessing error: {e}")

    # ===================== Model Training & Evaluation =====================
    st.header("Model Selection & Training")
    model_choice = st.selectbox("Choose Model",
                                ["Random Forest", "Logistic Regression", "SVM"])

    if 'X_train' in st.session_state and st.button("Train Selected Model"):
        try:
            if model_choice == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            elif model_choice == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            else:
                model = SVC()

            model.fit(st.session_state['X_train'], st.session_state['y_train'])
            st.success(f"{model_choice} trained successfully!")

            # === REAL EVALUATION ===
            y_pred = model.predict(st.session_state['X_val'])

            accuracy = accuracy_score(st.session_state['y_val'], y_pred)
            precision = precision_score(st.session_state['y_val'], y_pred, average='weighted')
            recall = recall_score(st.session_state['y_val'], y_pred, average='weighted')
            f1 = f1_score(st.session_state['y_val'], y_pred, average='weighted')

            st.subheader("Evaluation on Validation Data (REAL)")
            st.write(f"Accuracy: {accuracy:.2f}")
            st.write(f"Precision: {precision:.2f}")
            st.write(f"Recall: {recall:.2f}")
            st.write(f"F1-score: {f1:.2f}")

            report = classification_report(
                st.session_state['y_val'], y_pred, output_dict=True
            )
            st.dataframe(pd.DataFrame(report).transpose())

            # === Predict on external test ===
            test_predictions = model.predict(st.session_state['X_test'])
            result_df = st.session_state['test_df_original'].copy()
            result_df['Predicted_Survived'] = test_predictions

            st.subheader("Predictions on Test Data")
            st.write(result_df.head())

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Predictions CSV",
                csv,
                "titanic_predictions.csv"
            )

        except Exception as e:
            st.error(f"Training error: {e}")
