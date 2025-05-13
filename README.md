
# Heart Disease Prediction with Machine Learning

This project aims to predict the presence of heart disease using machine learning techniques. The dataset used contains patient information such as age, blood pressure, cholesterol levels, and other health-related features. A combination of preprocessing techniques and an ensemble model (Voting Classifier) was applied to achieve accurate predictions.

## Dataset

- Source: [Kaggle - Heart Disease Data](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
- Description: The dataset includes multiple features that may indicate the presence or absence of heart disease.

## Project Structure

### 1. Data Preprocessing

- Dropped unnecessary columns (`id`, `ca`)
- Handled missing values using median or mode
- Split data into training and testing sets
- Applied standardization for numerical features
- Applied one-hot encoding for categorical features

### 2. Model Building

- Used two models:
  - **Random Forest Classifier**
  - **K-Nearest Neighbors (KNN)**
- Combined them using a **Voting Classifier** (soft voting)

### 3. Evaluation

- Accuracy Score
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)

### 4. Visualization

- Pairplot of numeric features
- Class distribution plot
- Confusion matrix heatmap

## Results

The ensemble Voting Classifier achieved reliable performance on the test set, showing promising accuracy in predicting heart disease risk.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python heart_disease_prediction.py
   ```

## License

This project is licensed under the MIT License.
