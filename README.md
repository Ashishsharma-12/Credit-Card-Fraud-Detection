# Credit Card Fraud Detection

## Overview

This project implements a machine learning solution for detecting fraudulent credit card transactions, addressing a critical challenge in financial security. By leveraging advanced data science techniques, the project provides a robust model to identify potentially fraudulent activities with high accuracy.

## Key Features

- **Advanced Fraud Detection**: Utilizes Random Forest Classifier for reliable fraud identification
- **Robust Preprocessing**: Handles class imbalance using SMOTE
- **Comprehensive Analysis**: Includes feature importance and detailed model evaluation

## Dataset

The project uses the Credit Card Fraud Detection dataset from Kaggle, featuring:
- Transactions from European cardholders in September 2013
- Highly imbalanced dataset (fraudulent transactions: 0.172%)

## Project Performance

### Model Metrics
- **Accuracy**: 99.95%
- **Precision**: 85.71%
- **Recall**: 75.00%
- **F1-Score**: 80.00%

## Installation

### Prerequisites
- Python 3.8+
- Virtual environment recommended

### Setup Steps
1. Clone the repository
   ```bash
   git clone https://github.com/Ashishsharma-12/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
   ```

2. Create and activate a virtual environment
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Jupyter Notebook Workflow
1. Launch Jupyter Notebook
   ```bash
   jupyter notebook
   ```

2. Run notebooks in sequence:
   - `01_data_exploration.ipynb`: Dataset exploration
   - `02_data_preprocessing.ipynb`: Data preparation
   - `03_model_training.ipynb`: Model training
   - `04_model_evaluation.ipynb`: Performance evaluation

## Project Structure

```
Credit-Card-Fraud-Detection/
├── balanced.csv              # SMOTE-balanced dataset
├── code.ipynb                # Main data science workflow
├── conversation_dataset.json # Conversation data for LLM approach
├── LLM_based_fraud_detection.ipynb  # LLM fraud detection exploration
├── nearmiss.csv              # NearMiss undersampled dataset
├── README.md                 # Project documentation
└── test_dataset.json         # Test dataset
```

## Methodology

### Preprocessing Techniques
- **Class Imbalance**: SMOTE (Synthetic Minority Over-sampling Technique)
- **Feature Scaling**: StandardScaler normalization
- **Data Splitting**: Dedicated training and testing sets

### Model
- **Algorithm**: Random Forest Classifier
- **Strengths**: 
  - Robust handling of imbalanced datasets
  - Resistance to overfitting
  - Comprehensive feature importance analysis

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature-name`)
3. Commit changes
4. Push to the branch
5. Open a pull request

## License

MIT License

## Contact

- GitHub Issues: [Project Repository](https://github.com/Ashishsharma-12/Credit-Card-Fraud-Detection/issues)
- Email: ashishrsharma99@gmail.com

## Future Work

- Explore advanced LLM-based fraud detection techniques
- Implement real-time fraud detection capabilities
- Enhance model interpretability
