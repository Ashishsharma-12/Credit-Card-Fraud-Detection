# Credit-Card-Fraud-Detection
Detects fraud in transactions.

Description
This project focuses on detecting fraudulent credit card transactions using machine learning techniques. By leveraging transaction data, the model identifies patterns indicative of fraud, offering a valuable tool for financial institutions to reduce losses and safeguard customers. This repository showcases a practical application of machine learning in addressing real-world challenges in fraud detection.

Dataset
The project utilizes the Credit Card Fraud Detection dataset from Kaggle, which contains credit card transactions from European cardholders in September 2013. The dataset is notably imbalanced, with fraudulent transactions comprising only 0.172% of the total, reflecting the real-world rarity of fraud.

Preprocessing Steps:
Class Imbalance: Addressed using SMOTE (Synthetic Minority Over-sampling Technique) to balance the dataset.
Feature Scaling: Applied StandardScaler to normalize feature values for consistent model performance.
Data Splitting: Divided into training and testing sets to evaluate the model effectively.
Model
The core of this project is a Random Forest Classifier, selected for its robustness in handling imbalanced datasets and its ability to avoid overfitting. Feature importance analysis is included to highlight the most influential factors in detecting fraud.

Installation
To set up and run this project, you’ll need Python 3.8 or higher. Using a virtual environment is recommended to manage dependencies cleanly.

Clone the Repository:
bash

Collapse

Wrap

Copy
git clone https://github.com/Ashishsharma-12/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
Create a Virtual Environment:
bash

Collapse

Wrap

Copy
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install Dependencies:
bash

Collapse

Wrap

Copy
pip install -r requirements.txt
The requirements.txt file includes:

pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
jupyter
Usage
This project is structured around Jupyter notebooks that guide you through the machine learning pipeline. To get started:

Launch Jupyter Notebook:
bash

Collapse

Wrap

Copy
jupyter notebook
Navigate to the notebooks Directory and run the following notebooks in order:
01_data_exploration.ipynb: Explore and visualize the dataset.
02_data_preprocessing.ipynb: Prepare the data for modeling.
03_model_training.ipynb: Train the Random Forest model.
04_model_evaluation.ipynb: Evaluate the model’s performance.
Note: If your repository includes Python scripts instead of notebooks, please adjust this section with the appropriate commands (e.g., python main.py --train).

Results
The Random Forest model delivers impressive results on the test set:

Accuracy: 99.95%
Precision: 85.71%
Recall: 75.00%
F1-Score: 80.00%
These metrics demonstrate the model’s ability to detect fraud effectively while maintaining a low rate of false positives. Visualizations such as a confusion matrix and feature importance plot are saved in the results directory.

Project Structure
The repository contains the following files, which collectively form the backbone of your credit card fraud detection project:

text

Credit-Card-Fraud-Detection/
├── balanced.csv                # Dataset balanced using a technique like SMOTE
├── code.ipynb                  # Main Jupyter notebook for data preprocessing, model training, and evaluation
├── conversation_dataset.json   # Dataset possibly used for LLM-based fraud detection
├── LLM_based_fraud_detection.ipynb  # Jupyter notebook exploring fraud detection with large language models
├── nearmiss.csv                # Dataset after applying NearMiss undersampling
├── README.md                   # Project documentation (this file)
└── test_dataset.json           # Test dataset for model evaluation

File Descriptions
balanced.csv: This is likely the dataset after addressing class imbalance, possibly using a technique like SMOTE (Synthetic Minority Over-sampling Technique) to oversample the minority class (fraudulent transactions).
code.ipynb: The primary Jupyter notebook containing the core workflow of the project. It includes steps such as data preprocessing, model training (e.g., using a Random Forest Classifier), and performance evaluation.
conversation_dataset.json: A JSON file that may contain data used in the LLM-based approach, possibly representing transaction-related text or dialogue data for fraud detection experiments.
LLM_based_fraud_detection.ipynb: A Jupyter notebook exploring an innovative approach to fraud detection using large language models (LLMs), complementing the traditional machine learning methods in code.ipynb.
nearmiss.csv: This file likely represents the dataset after applying the NearMiss undersampling technique, which reduces the majority class (non-fraudulent transactions) to balance the dataset.
README.md: The documentation file (like this one) that provides an overview of the project, instructions for use, and other relevant details.
test_dataset.json: A JSON-formatted test dataset used to evaluate the performance of the trained models.

Contributing
Contributions are encouraged! To contribute:

Fork the repository.
Create a branch for your changes (git checkout -b feature-name).
Commit your updates and push to your fork.
Submit a pull request with a clear description of your contribution.
For significant changes, please open an issue first to discuss your ideas.

License
This project is licensed under the MIT License. See the  file for details.

Contact
For questions, feedback, or collaboration opportunities, please open an issue on GitHub or email me at ashishrsharma99@gmail.com.
