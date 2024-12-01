# Alzheimer's Disease Prediction Using EEG Data

This project aims to classify Alzheimer’s Disease (AD) presence using EEG band power features. It compares multiple machine learning models (Random Forest, XGBoost, and SVM) to predict AD status based on processed EEG data.

## Project Overview

### Data
The dataset used in this project can be found [here on OSF](https://osf.io/s74qf/). It consists of EEG recordings from four groups:
- **Group A**: Healthy elderly subjects, eyes open
- **Group B**: Healthy elderly subjects, eyes closed
- **Group C**: Probable AD patients, eyes open
- **Group D**: Probable AD patients, eyes closed

Each EEG recording is transformed into features representing power in delta, theta, alpha, and beta frequency bands, with additional band ratios added for enhanced prediction.

### Workflow
1. **Data Preprocessing**: Band power calculation and feature engineering to add band ratios.
2. **Modeling**: 
    - Use of pipelines to prevent data leakage and streamline preprocessing.
    - Three models are compared using cross-validation: Random Forest, XGBoost, and SVM (with RBF kernel).
3. **Evaluation**:
    - Cross-validation to select the best-performing model.
    - Final evaluation on a held-out test set, with a confusion matrix visualization.

## Requirements

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Project Structure

- **data/**: Folder containing raw EEG data files.
- **notebooks/**: Jupyter notebooks for each step in the analysis.
- **src/**: Source code for feature engineering, model building, and evaluation.
- **README.md**: Project documentation.
- **requirements.txt**: Dependencies for reproducibility.

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/alzheimers-eeg-prediction.git
    cd alzheimers-eeg-prediction
    ```

2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the main analysis script or open the notebooks in `notebooks/` for interactive exploration.

## Analysis Steps

### 1. Data Preprocessing
Each EEG signal undergoes:
- Band-pass filtering to isolate delta, theta, alpha, and beta frequencies.
- Power calculation for each band, with additional ratios like theta/alpha and theta/beta for enhanced feature representation.

### 2. Model Building
Three models were tested:
- **Random Forest** (no scaling needed).
- **XGBoost** and **SVM**: Both use pipelines with standard scaling to ensure consistent performance.

### 3. Evaluation
Models are evaluated using 5-fold cross-validation, and the best-performing model is selected for final evaluation on a held-out test set. The final model’s performance is visualized using a confusion matrix.

## Results
The project concludes with the best model’s accuracy on the test set, confusion matrix visualization, and detailed evaluation metrics.

## License
This project is licensed under the MIT License.

## Acknowledgments
Special thanks to the Alzheimer's research community for providing guidance on EEG band analysis for AD prediction.

