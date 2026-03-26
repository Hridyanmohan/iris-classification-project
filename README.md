🌼 Iris Classification Dashboard

An interactive Machine Learning web application built using Dash and Scikit-learn to classify Iris flower species based on selected input features.

Project Overview
This project demonstrates an end-to-end ML workflow integrated with a web dashboard.
Users can dynamically select features, train a model in real-time, and visualize predictions along with performance metrics.

🚀 Features
- Dynamic feature selection using dropdown
- Real-time model training (Random Forest Classifier)
- Interactive data visualization (Plotly)
- Accuracy display
- Confusion Matrix visualization
- Classification Report (Precision, Recall, F1-score)
- Machine Learning Details
Dataset: Iris Dataset (built-in from Scikit-learn)
Model: Random Forest Classifier
Evaluation Metrics:
Accuracy Score
Confusion Matrix
Precision, Recall, F1-score

📊 Key Insights
🌼 Setosa is easily separable and highly accurate
🌿 Versicolor & Virginica show overlapping features
📉 Accuracy varies depending on selected features
🔍 Feature selection significantly impacts model performance


🛠️ Tech Stack
Python
Dash
Plotly
Pandas
Scikit-learn
NumPy

📁 Project Structure
IRIS CLASSIFICATION PROJECT/
│
├── app.py
├── requirements.txt
└── README.md


⚙️ Installation & Setup
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/iris-classification-project.git
cd iris-classification-project
2. Install dependencies
pip install -r requirements.txt
3. Run the application
python classificationapp.py


🌐 Usage
Select input features from the dropdown
View predictions in graphical format
Analyze model performance using:
Accuracy score
Confusion matrix
Classification report
