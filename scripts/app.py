import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
from io import BytesIO
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC

from sklearn.metrics import f1_score

def train_model(df, target_column, model, le=None):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    if le is not None:
        y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_score = accuracy_score(y_train, y_train_pred)
    test_score = accuracy_score(y_test, y_test_pred)

    return model, (train_score, test_score), le

st.set_page_config(page_title='ML Web Application', page_icon=':books:', layout='wide')

st.sidebar.title('Navigation')
page = st.sidebar.radio('', ['Train', 'Deploy'])

if page == 'Train':
    st.header('Train a Machine Learning Model')
    file = st.file_uploader('Upload your data as a csv file', type='csv')

    if file is not None:
        df = pd.read_csv(file)
        st.dataframe(df)

        drop_cols = st.multiselect('Select the columns to drop', df.columns.tolist())
        if drop_cols:
            df = df.drop(drop_cols, axis=1)

        target_col = st.selectbox('Select the target column', df.columns.tolist())
        df[target_col] = df[target_col].astype('category')
        is_cat = df[target_col].nunique() < 4
        le = None
        if is_cat:
            st.subheader('Classification Problem')
            model_choice = st.selectbox('Select the model', ['Decision Tree', 'Logistic Regression', 'Support Vector Machines'])
            model_dict = {'Decision Tree': DecisionTreeClassifier(), 'Logistic Regression': LogisticRegression(), 'Support Vector Machines': SVC()}
            le = LabelEncoder()
        else:
            st.subheader('Regression Problem')
            model_choice = st.selectbox('Select the model', ['Ridge Regression', 'KNN', 'Decision Tree Regressor'])
            model_dict = {'Ridge Regression': Ridge(), 'KNN': KNeighborsRegressor(), 'Decision Tree Regressor': DecisionTreeRegressor()}

        model = model_dict[model_choice]

        num_cols = df.drop(target_col, axis=1).select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.drop(target_col, axis=1).select_dtypes(include='category').columns.tolist()

        num_pipe = Pipeline([('scaler', StandardScaler())])
        cat_pipe = Pipeline([('encoder', OneHotEncoder())])

        preprocessor = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])

        full_pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])

        model, scores, le = train_model(df, target_col, full_pipeline, le)

        st.write('Train Accuracy: ', scores[0])
        st.write('Test Accuracy: ', scores[1])

        download_file = st.button('Download Pickle File')

        if download_file:
            pickle_data = pickle.dumps((model, preprocessor, target_col, le))
            b64_pickle = base64.b64encode(pickle_data).decode()  
            href = f'<a href="data:application/octet-stream;base64,{b64_pickle}" download="model.pkl">Download Pickle File</a>'
            st.markdown(href, unsafe_allow_html=True)

elif page == 'Deploy':
    st.header('Deploy a Machine Learning Model')
    file_model = st.file_uploader('Upload your Pickle file', type=['pkl'])

    if file_model is not None:
        model, preprocessor, target_col, le = pickle.load(BytesIO(file_model.read()))
        st.write('Model and Transformer loaded successfully!')

        try:
            st.write('Feature names: ', preprocessor.get_feature_names_out())
        except:
            st.write('Cannot retrieve feature names!')

        file = st.file_uploader('Upload your data as a csv file', type='csv')
        if file is not None:
            df_deploy = pd.read_csv(file)
            if target_col in df_deploy.columns:
                df_deploy = df_deploy.drop(target_col, axis=1)
            predictions = model.predict(df_deploy)
            if le is not None:
                predictions = le.inverse_transform(predictions)
            df_deploy['Predicted_target'] = predictions
            st.dataframe(df_deploy)

            download_file = st.button('Download CSV with Predictions')
            if download_file:
                csv = df_deploy.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # encoding the csv file to base64
                href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">Download csv file</a>'
                st.markdown(href, unsafe_allow_html=True)
