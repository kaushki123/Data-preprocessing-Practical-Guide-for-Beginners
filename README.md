# Data-preprocessing-Practical-Guide-for-Beginners
Data Preprocessing: Titanic Dataset Python pipeline cleans raw data:  Missing values: Median Age, mode Embar.ked, drop Cabin  Encode Sex/Embarked  Scale Age/Fare  Feature selection via correlation  80/20 train-test split Output: Clean 891×11 dataset for reliable modeling. Transform data for accurate ML predictions.

# Data Preprocessing: Titanic Dataset
## Overview
This project demonstrates a complete data preprocessing pipeline for the [Titanic dataset](https://www.kaggle.com/c/titanic). The workflow transforms raw data into a model-ready format through cleaning, encoding, scaling, and feature selection.

## Workflow Steps
1. **Data Collection**  
   ```python
   import pandas as pd
   df = pd.read_csv("titanic.csv")
   
   
2. **Data Cleaning**
    * Fill missing Age with median:
      df['Age'].fillna(df['Age'].median(), inplace=True)

    * Drop Cabin column:
      df.drop('Cabin', axis=1, inplace=True)

    * Remove duplicates:
      df.drop_duplicates(inplace=True)

3.  **Exploratory Data Analysis (EDA)**

     * Check dataset structure:
       df.shape, df.info(), df.describe()

      * Visualize correlations:
        import seaborn as sns
        sns.heatmap(df.corr(), annot=True)

 4. **Feature Engineering**
    * Encode Sex (Label Encoding):
      from sklearn.preprocessing import LabelEncoder
      df['Sex'] = LabelEncoder().fit_transform(df['Sex'])

    *  One-hot encode Embarked:
      df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

5. **Feature Scaling**
   from sklearn.preprocessing import StandardScaler
df[['Age','Fare']] = StandardScaler().fit_transform(df[['Age','Fare']])

6. **Feature Selection**

   Select features with >0.2 correlation to target (Survived):
   features = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked_Q','Embarked_S']

7. **Data Splitting**
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
    df[features], df['Survived'], test_size=0.2, random_state=42
     )

**Results**

**Dataset Size**: 891 rows × 11 features after preprocessing

**Missing Values**: 0

**Training/Test Split**: 712 / 179 samples

 **Top Correlated Features**:
 
   * Sex ➔ 0.54
    
   * Pclass ➔ -0.34

**Dependencies**
  * Python 3.8+
  * Libraries:
     * pandas==1.4.2
     * scikit-learn==1.1.1
     * seaborn==0.11.2

**Usage**

Run the Jupyter notebook titanic_preprocessing.ipynb to execute the full workflow.

This GitHub-ready structure includes:  
- Concise description for repository overview  
- Detailed README.md with code snippets and explanations  
- Clear workflow visualization and dependency instructions  
- Emphasis on reproducible preprocessing steps

