ChurnGuard - Customer Churn Prediction System

A machine learning-powered web application for predicting customer churn and supporting retention strategies.

Overview:
ChurnGuard is a web-based customer churn prediction system developed as part of my machine learning and web development portfolio. This application demonstrates the integration of machine learning models with user-friendly web interfaces to solve real business problems.

*Problem Statement: Customer churn is a critical business challenge where companies lose customers over time. Early prediction of at-risk customers can help businesses implement targeted retention strategies.

*Solution: A Flask web application that uses machine learning to predict churn probability and provides actionable insights through an interactive dashboard.

Technologies Used
*Backend Development:

 1. Flask - Web framework and routing
 2. Python - Core programming language

*Machine Learning:

 1. Scikit-learn - Customer churn prediction models
 2. Pandas - Data manipulation and analysis
 3. NumPy - Numerical computations
 4. Joblib - Model persistence and serialization

*Data Visualization:

 1. Plotly - Interactive charts and graphs
 2. Matplotlib - Additional plotting capabilities

*Frontend Technologies:

 1. HTML5 - Structure and content
 2. CSS3 - Styling and responsive design
 3. JavaScript - Dynamic interactions and chart rendering

*Development Tools:

 1. Git - Version control
 2. VS Code - Development environment
 3. Virtual Environment - Dependency management

Model Information

*Dataset Features
 The model analyzes customer data across multiple dimensions:

   - Service Usage: Phone, internet, and streaming services
   - Contract Details: Contract length and payment preferences
   - Financial Data: Monthly and total charges
   - Demographics: Age, gender, and tenure
   - Support Interaction: Technical support usage patterns

*Performance Metrics

   - Accuracy: Trained and validated on telecom customer data
   - Response Time: Real-time predictions (<2 seconds)
   - Feature Engineering: Optimized input preprocessing
   - Model Type: Classification model for binary churn prediction


User Navigation:

 - Welcome Page: Start at the welcome page and click "Get Started."
   
<img width="1908" height="964" alt="image" src="https://github.com/user-attachments/assets/3dc8a40f-ff16-4a1a-b062-abaf0fc3a7bc" />

 - Login/Signup: Sign up with a new account or log in to access the dashboard.
   
<img width="1893" height="968" alt="image" src="https://github.com/user-attachments/assets/34d75c48-de25-4b26-b2bd-1ffd17a0f859" />

- Dashboard: On the dashboard, fill out the form with the customer's data, including demographics, service subscriptions, and billing information.
  
<img width="1895" height="969" alt="image" src="https://github.com/user-attachments/assets/4c69e7d2-f537-4f9e-974e-4d1218ea0e99" />

<img width="1887" height="959" alt="image" src="https://github.com/user-attachments/assets/beac4974-9b37-4f50-bbd9-a424be5348da" />

<img width="1896" height="954" alt="image" src="https://github.com/user-attachments/assets/fad9b61d-25b4-488d-88be-b6f402735381" />

- Predict Churn: Click the "Predict Churn" button. The application will process the data and display the results directly on the same page.
  
<img width="1897" height="962" alt="image" src="https://github.com/user-attachments/assets/6c58458f-c583-4f7e-857f-3c7b3916fdc7" />

Future Improvements:

-Model Accuracy: Experiment with different algorithms (Random Forest, XGBoost)
-Batch Processing: Upload CSV files for bulk customer analysis
-Export Features: PDF reports and data export capabilities










