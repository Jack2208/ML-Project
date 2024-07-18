import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def feature_creation(df : pd.DataFrame) -> pd.DataFrame:
    df_raw.dropna(inplace=True)
    df['password'] = df['password'].astype(str)
    df['length'] = df['password'].apply(len)
    df['caps_alpbts'] = df['password'].apply(lambda x : len(re.findall('[A-Z]', x)))/df['length']
    df['small_alpbts'] = df['password'].apply(lambda x : len(re.findall('[a-z]', x)))/df['length']
    df['num'] = df['password'].apply(lambda x : len(re.findall('[0-9]', x)))/df['length']
    df['comon_chars'] = df['password'].apply(lambda x : len(re.findall('[@_!#$%^&*()<>?/\\|{ }~:\\[\\]]', x)))/df['length']
    df['unique_chars'] = df['password'].apply(lambda x : len(re.findall('[^a-zA-Z0-9@_!#$%^&*()+-<>?/\|{ }~:\\[\\]]', x)))/df['length']
    return df


dataset = 'data/passwords.csv'
df_raw = pd.read_csv(dataset, on_bad_lines='skip')
df_raw.dropna(inplace=True)

df = feature_creation(df_raw)

x_temp = np.array(df["password"])
y = np.array(df["strength"])

# Trasformazione TF-IDF
tfidf = TfidfVectorizer(analyzer="char", lowercase=False, token_pattern=None)
x_temp = tfidf.fit_transform(x_temp).toarray()

print('Shape x_temp :', x_temp.shape)
print('Shape df :', df.iloc[:,3:].shape)

x = np.concatenate((x_temp, df.iloc[:,3:].values), axis=1)
print(x.shape)


# Combinazione delle caratteristiche TF-IDF con le caratteristiche manuali
print(np.concatenate((tfidf.get_feature_names_out(), df.columns[3:])))

print(list(df.columns[3:]))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
rf_model = RandomForestClassifier(n_jobs=-1)
rf_model.fit(x_train, y_train)
rf_pred = rf_model.predict(x_test)
print("Score of the model is", rf_model.score(x_test, y_test), ".")
print(classification_report(y_test, rf_pred))


logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(x_train, y_train)
logistic_pred = logistic_model.predict(x_test)

logistic_model.score(x_test, y_test)

print(classification_report(y_test, logistic_pred))