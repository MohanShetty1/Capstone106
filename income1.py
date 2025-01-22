import sqlite3
from flask import Flask, request, jsonify, render_template, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix
import joblib

app = Flask(__name__)

# Step 1: Connect to SQLite Database and Create Table
def init_db():
    conn = sqlite3.connect('tax_data.db')  # Database file
    cursor = conn.cursor()
    
    # Create table to store user data
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tax_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT,
            last_name TEXT,
            income REAL,
            deductions REAL,
            expenses REAL,
            tax_paid REAL,
            is_fraud INTEGER,
            amount_to_be_paid REAL
        )
    ''')
    conn.commit()
    conn.close()

# Initialize database
init_db()

# Step 2: Function to calculate tax based on tax slabs
def calculate_tax(taxable_income):
    tax = 0
    
    if taxable_income <= 100000:
        # 0% tax for income <= 100000
        tax = 0
    elif taxable_income <= 300000:
        # 5% tax for income between 100001 and 300000
        tax = (taxable_income - 100000) * 0.05
    elif taxable_income <= 700000:
        # 10% tax for income between 300001 and 700000
        tax = (200000 * 0.05) + (taxable_income - 300000) * 0.10
    else:
        # 15% tax for income above 700000
        tax = (200000 * 0.05) + (400000 * 0.10) + (taxable_income - 700000) * 0.15
    
    return tax


# Step 3: Train Random Forest Classifier using a real dataset
def train_model():
    # Load the real dataset from tax.csv
    data = pd.read_csv('tax.csv')

    # Calculate taxable income and expected tax
    data['taxable_income'] = data['income'] - data['deductions']
    data['expected_tax'] = data['taxable_income'].apply(calculate_tax)

    # Label fraud (adjust logic as needed)
    data['is_fraud'] = (data['tax_paid'] < data['expected_tax'] * 0.8).astype(int)

    # Prepare features (X) and labels (y)
    X = data[['income', 'deductions', 'expenses', 'tax_paid', 'expected_tax']]
    y = data['is_fraud']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the Random Forest Classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Save the trained model to a file
    joblib.dump(clf, 'fraud_detection_model.pkl')

    # Make predictions on the test data
    y_pred = clf.predict(X_test)

    # Calculate accuracy, precision, recall, F1-score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Print evaluation metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return clf

# Train the model when the app starts
model = train_model()

@app.route('/')
def home():
    return render_template('index.html')

# Step 4: API endpoint to predict fraud and save data to the database
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    first_name = data['first_name']
    last_name = data['last_name']
    income = float(data['income'])
    deductions = float(data['deductions'])
    expenses = float(data['expenses'])
    tax_paid = float(data['tax_paid'])

    # Calculate taxable income and expected tax
    taxable_income = income - deductions
    expected_tax = calculate_tax(taxable_income)

    # Prepare data for the model
    input_data = pd.DataFrame({
        'income': [income],
        'deductions': [deductions],
        'expenses': [expenses],
        'tax_paid': [tax_paid],
        'expected_tax': [expected_tax]
    })

    # Predict fraud
    is_fraud = model.predict(input_data)[0]
    amount_to_be_paid = 0
    if is_fraud:
        amount_to_be_paid = expected_tax - tax_paid

    # Insert data into the database
    conn = sqlite3.connect('tax_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO tax_records (first_name, last_name, income, deductions, expenses, tax_paid, is_fraud, amount_to_be_paid)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (first_name, last_name, income, deductions, expenses, tax_paid, int(is_fraud), amount_to_be_paid))
    conn.commit()
    conn.close()

    # Return the prediction result
    return jsonify({
        'first_name': first_name,
        'last_name': last_name,
        'is_fraud': bool(is_fraud),
        'amount_to_be_paid': round(amount_to_be_paid, 2)
    })
@app.route('/confusion-matrix', methods=['GET'])
def generate_confusion_matrix():
    # Load saved confusion data
    confusion_data = pd.read_csv('confusion_data.csv')
    y_test = confusion_data['y_test']
    y_pred = confusion_data['y_pred']

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot and save confusion matrix as an image
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig('confusion_matrix.png')

    # Return the saved image
    return send_file('confusion_matrix.png', mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
