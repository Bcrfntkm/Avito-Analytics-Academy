from collections import Counter

class CountVectorizer:
    def __init__(self):
        self.features = []

    def fit_transform(self, corpus):
        self.features = []
        self.result = [[] for _ in range(len(corpus))]
        for text in corpus:
            for word in text.lower().split():
                if word not in self.features:
                    self.features.append(word) 
        for i, text in enumerate(corpus):
            counter = Counter(text.lower().split())
            for word in self.features:
                self.result[i].append(counter[word])
        return self.result
        
    def get_feature_names(self):
        return list(self.features)