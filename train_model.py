from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pickle

texts = [
    "The government announced new education reforms",
    "Celebrity says earth is flat and aliens live among us",
    "Scientists discover new species in Amazon",
    "Miracle cure guarantees weight loss in 2 days"
]

labels = [1, 0, 1, 0]  # 1 = Real, 0 = Fake

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])

model.fit(texts, labels)

with open("fake_news_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved.")
