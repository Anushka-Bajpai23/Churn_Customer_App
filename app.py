from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Change this to a secure key

# Load the trained model and preprocessor
try:
    model = joblib.load('model/churn_model.pkl')
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: Model file 'churn_model.pkl' not found. Please run train_model.py first.")
    exit()

# List of features in the correct order for the model
feature_order = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 
                 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                 'Contract', 'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges', 'TotalCharges']

def preprocess_input(data):
    """Preprocesses a single user input for prediction."""
    df_input = pd.DataFrame([data], columns=feature_order)
    df_input['TotalCharges'] = pd.to_numeric(df_input['TotalCharges'], errors='coerce')
    df_input['TotalCharges'] = df_input['TotalCharges'].fillna(0) # or another suitable value
    df_input['SeniorCitizen'] = df_input['SeniorCitizen'].astype(int)
    df_input['tenure'] = df_input['tenure'].astype(int)
    df_input['MonthlyCharges'] = df_input['MonthlyCharges'].astype(float)
    return df_input

def get_insights(churn_prob):
    """Provides insights and alerts based on churn probability."""
    if churn_prob > 0.7:
        return {
            'risk': 'High Risk 🔴',
            'message': "This customer has a high probability of churning. Immediate action is required.",
            'actions': "Offer a personalized retention plan, provide a discount, or connect with a senior support agent."
        }
    elif churn_prob > 0.4:
        return {
            'risk': 'Medium Risk 🟡',
            'message': "This customer shows signs of potential churn. Proactive engagement can prevent it.",
            'actions': "Send a targeted email campaign highlighting new services or provide a small loyalty bonus."
        }
    else:
        return {
            'risk': 'Low Risk 🟢',
            'message': "The customer is likely to stay. Continue with standard service and monitor for changes.",
            'actions': "Ensure continued good service and positive customer experiences."
        }

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Simple placeholder login logic
        session['logged_in'] = True
        return redirect(url_for('dashboard'))
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Simple placeholder signup logic
        session['logged_in'] = True
        return redirect(url_for('dashboard'))
    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401

    try:
        user_input = {key: request.form[key] for key in feature_order}
        
        # Preprocess the input data
        processed_input = preprocess_input(user_input)
        
        # Make a prediction
        churn_prob = model.predict_proba(processed_input)[:, 1][0]
        churn_pred = 'Yes' if churn_prob > 0.5 else 'No'
        
        # Generate dynamic charts
        # Chart 1: Churn probability gauge
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_prob * 100,
            title={'text': "Churn Probability"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "#00AEEF"},
                   'steps': [
                       {'range': [0, 40], 'color': "green"},
                       {'range': [40, 70], 'color': "yellow"},
                       {'range': [70, 100], 'color': "red"}],
                   'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': churn_prob * 100}}))
        gauge_fig.update_layout(height=250, margin=dict(l=20, r=20, t=30, b=20))
        gauge_json = gauge_fig.to_json()
        
        # Chart 2: Top factors contributing to prediction (dummy data for now)
        # In a real-world app, you would use a model-agnostic interpreter like SHAP or LIME
        # to get the actual feature importances for this specific prediction.
        # Here we'll use a simplified logic based on some of the inputs.
        
        factors = {
            'Monthly Charges': float(user_input['MonthlyCharges']),
            'Tenure': int(user_input['tenure']),
            'Contract': 1 if user_input['Contract'] == 'Month-to-month' else 0,
            'Internet Service': 1 if user_input['InternetService'] == 'Fiber optic' else 0,
            'Senior Citizen': 1 if user_input['SeniorCitizen'] == '1' else 0
        }
        
        sorted_factors = sorted(factors.items(), key=lambda item: item[1], reverse=True)
        factor_names = [f[0] for f in sorted_factors]
        factor_values = [f[1] for f in sorted_factors]
        
        bar_fig = px.bar(
            x=factor_names,
            y=factor_values,
            labels={'x': 'Factor', 'y': 'Relative Impact'},
            title='Top Contributing Factors to Churn'
        )
        bar_fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=20))
        bar_json = bar_fig.to_json()
        
        # Get necessary alerts and insights
        insights = get_insights(churn_prob)

        return jsonify({
            'churn_prob': round(churn_prob, 2),
            'churn_pred': churn_pred,
            'gauge_chart': gauge_json,
            'bar_chart': bar_json,
            'insights': insights
        })

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
