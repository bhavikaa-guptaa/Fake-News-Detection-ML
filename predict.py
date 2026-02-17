import pickle

with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)

text = input("Enter news text: ")
prediction = model.predict([text])

if prediction[0] == 1:
    print("Prediction: REAL news")
else:
    print("Prediction: FAKE news")
