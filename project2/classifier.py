import joblib
from sklearn.linear_model import LogisticRegression

class TextClassifier:
    def __init__(self, vectorizer, clf=None):
        self.vectorizer = vectorizer
        self.clf = clf or LogisticRegression(max_iter=500)

    def fit(self, texts, labels):
        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.clf.predict(X)

    def score(self, texts, labels):
        X = self.vectorizer.transform(texts)
        return self.clf.score(X, labels)

    def save(self, vec_path, clf_path):
        joblib.dump(self.vectorizer, vec_path)
        joblib.dump(self.clf, clf_path)

    @classmethod
    def load(cls, vec_path, clf_path):
        vec = joblib.load(vec_path)
        clf = joblib.load(clf_path)
        return cls(vec, clf)
