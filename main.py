import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def feature_creation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to create new features from the password dataset.
    """
    df['password'] = df['password'].astype(str)
    df['length'] = df['password'].apply(len)
    # Calculate the proportion of uppercase letters, lowercase letters, numbers and special characters in each password
    df['caps_alphabets'] = df['password'].apply(lambda x: len(re.findall('[A-Z]', x))) / df['length']
    df['small_alphabets'] = df['password'].apply(lambda x: len(re.findall('[a-z]', x))) / df['length']
    df['num'] = df['password'].apply(lambda x: len(re.findall('[0-9]', x))) / df['length']
    df['common_chars'] = df['password'].apply(lambda x: len(re.findall('[@_!#$%^&*()<>?/\\|{}~:\\[\\]]', x))) / df['length']
    df['unique_chars'] = df['password'].apply(lambda x: len(re.findall('[^a-zA-Z0-9@_!#$%^&*()+-<>?/\|{}~:\\[\\]]', x))) / df['length']
    return df


dataset_path = 'data/passwords.csv'
# Read the dataset, skip bad lines and drop missing values
df_raw = pd.read_csv(dataset_path, on_bad_lines='skip').dropna()

df = feature_creation(df_raw)

x_temp = np.array(df["password"])
y = np.array(df["strength"])

# Transform the passwords into TF-IDF features
tfidf = TfidfVectorizer(analyzer="char", lowercase=False, token_pattern=None)
x_tfidf = tfidf.fit_transform(x_temp).toarray()

# Test prints to understand the shapes of the data
# print('Shape of x_tfidf:', x_tfidf.shape)
# print('Shape of manual features:', df.iloc[:, 3:].shape)

# Concatenate TF-IDF features with the manually created features
x = np.concatenate((x_tfidf, df.iloc[:, 3:].values), axis=1)
# print('Shape of combined features:', x.shape)

# Combine the TF-IDF feature names with the manual feature names
feature_names = np.concatenate((tfidf.get_feature_names_out(), df.columns[3:]))
# print('Feature names:', feature_names)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Train a RandomForestClassifier model
rf_model = RandomForestClassifier(n_jobs=-1)
rf_model.fit(x_train, y_train)

# Predict the test set results
rf_pred = rf_model.predict(x_test)

print("Random Forest Model Score:", rf_model.score(x_test, y_test))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))

# Confusion matrix for RandomForestClassifier
rf_cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Weak', 'Medium', 'Strong'], yticklabels=['Weak', 'Medium', 'Strong'])
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Train a LogisticRegression model
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(x_train, y_train)

# Predict the test set results
logistic_pred = logistic_model.predict(x_test)

print("Logistic Regression Model Score:", logistic_model.score(x_test, y_test))
print("Logistic Regression Classification Report:\n", classification_report(y_test, logistic_pred))

# Confusion matrix for LogisticRegression
logistic_cm = confusion_matrix(y_test, logistic_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(logistic_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Weak', 'Medium', 'Strong'], yticklabels=['Weak', 'Medium', 'Strong'])
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()