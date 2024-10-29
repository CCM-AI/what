import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from flask import Flask, jsonify, request, render_template, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from flask_bootstrap import Bootstrap
from cryptography.fernet import Fernet
import logging

# Logging Configuration
logging.basicConfig(level=logging.INFO)

# Flask App Setup
app = Flask(__name__)
Bootstrap(app)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chronic_care.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_secret_key'  # Change this to a random secret key
db = SQLAlchemy(app)

# Security Setup
key = Fernet.generate_key()
cipher = Fernet(key)

# Models
class PatientRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Integer)
    gender = db.Column(db.String(10))
    chronic_conditions = db.Column(db.String(200))
    lifestyle_factors = db.Column(db.String(100))
    risk = db.Column(db.Integer)
    care_plan = db.Column(db.String(200))
    self_management_support = db.Column(db.String(200))
    follow_up_schedule = db.Column(db.String(200))
    outcome = db.Column(db.String(50))
    quality_improvement_feedback = db.Column(db.String(200))

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)

# Create all database tables
with app.app_context():
    db.create_all()

# Step 1: Patient Identification
def identify_patients():
    # For demonstration purposes, we use dummy data.
    data = {
        'id': [1, 2, 3],
        'age': [65, 72, 58],
        'gender': ['M', 'F', 'F'],
        'chronic_conditions': ['Diabetes', 'Hypertension', 'Heart Disease'],
        'lifestyle_factors': ['Sedentary', 'Active', 'Sedentary']
    }
    return pd.DataFrame(data)

# Step 2: AI-driven Risk Stratification
def risk_stratification(patient_data):
    X = patient_data[['age']]  # You should expand this with more features.
    y = np.array([0, 1, 1])  # Dummy risk labels for illustration purposes.

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Grid search for hyperparameter tuning
    params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
    }
    grid_search = GridSearchCV(RandomForestClassifier(), params, cv=5)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logging.info(f"Best model parameters: {grid_search.best_params_}")

    predictions = best_model.predict(X_test)
    logging.info(f"Classification Report:\n {classification_report(y_test, predictions)}")
    
    patient_data['risk'] = best_model.predict(X)
    return patient_data

# Step 3: Personalized Care Plans
def create_care_plans(patient_data):
    care_plans = []
    for index, patient in patient_data.iterrows():
        if patient['risk'] == 1:  # High risk
            plan = f"Intensive monitoring and medication management for {patient['id']}"
        else:  # Low risk
            plan = f"Regular check-ups and lifestyle counseling for {patient['id']}"
        care_plans.append(plan)
    patient_data['care_plan'] = care_plans
    return patient_data

# Step 4: Self-management Support
def self_management_support(patient_data):
    support_resources = []
    for index, patient in patient_data.iterrows():
        if patient['risk'] == 1:
            resource = "Access to 24/7 telehealth services and specialized nutrition counseling."
        else:
            resource = "Regular health newsletters and community exercise programs."
        support_resources.append(resource)
    patient_data['self_management_support'] = support_resources
    return patient_data

# Step 5: Monitoring & Follow-up
def monitoring_follow_up(patient_data):
    follow_up_schedule = []
    for index, patient in patient_data.iterrows():
        schedule = f"Follow-up in 1 month for {patient['id']}"
        follow_up_schedule.append(schedule)
    patient_data['follow_up_schedule'] = follow_up_schedule
    return patient_data

# Step 6: Outcome Evaluation
def evaluate_outcomes(patient_data):
    outcomes = []
    for index, patient in patient_data.iterrows():
        outcome = np.random.choice(['Improved', 'Stable', 'Worsened'])  # Dummy outcomes
        outcomes.append(outcome)
    patient_data['outcome'] = outcomes
    return patient_data

# Step 7: Quality Improvement
def quality_improvement(patient_data):
    improvement_feedback = []
    for index, patient in patient_data.iterrows():
        if patient['outcome'] == 'Improved':
            feedback = "Continue current care plan."
        else:
            feedback = "Reassess care plan for potential adjustments."
        improvement_feedback.append(feedback)
    patient_data['quality_improvement_feedback'] = improvement_feedback
    return patient_data

# User Registration
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        new_user = User(username=username, password=password)
        db.session.add(new_user)
        db.session.commit()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')

# User Login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username, password=password).first()
        if user:
            session['user_id'] = user.id
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Login failed. Check your username and/or password.', 'danger')
    return render_template('login.html')

# Patient Dashboard
@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    # Retrieve patient data for the logged-in user
    patient_data = identify_patients()  # Placeholder for actual patient data retrieval
    patient_data = risk_stratification(patient_data)
    patient_data = create_care_plans(patient_data)
    patient_data = self_management_support(patient_data)
    patient_data = monitoring_follow_up(patient_data)
    patient_data = evaluate_outcomes(patient_data)
    patient_data = quality_improvement(patient_data)

    return render_template('dashboard.html', patients=patient_data)

# Patient Logout
@app.route('/logout')
def logout():
    session.pop('user_id', None)
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# Main Execution Flow
def main():
    db.create_all()  # Ensure the database and tables are created
    app.run(debug=True)

if __name__ == "__main__":
    main()
