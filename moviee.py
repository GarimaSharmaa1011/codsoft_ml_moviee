import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline


nltk.download('stopwords')
from nltk.corpus import stopwords # type: ignore
stop_words = stopwords.words('english')


df = pd.read_csv('movie_data.csv')  

def preprocess_text(text):
    # Remove non-alphabetic characters
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    # Convert text to lower case
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text


df['plot_summary'] = df['plot_summary'].apply(preprocess_text)


X = df['plot_summary']
y = df['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = make_pipeline(
    TfidfVectorizer(), 
    LogisticRegression(max_iter=1000)  
)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


print(classification_report(y_test, y_pred))


sample_plot = "A scientist creates a dangerous experiment that goes wrong."
predicted_genre = model.predict([sample_plot])[0]
print(f"Predicted Genre: {predicted_genre}")
