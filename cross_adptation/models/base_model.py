from abc import ABC



class AdaptationClassifier(ABC):

    def fit(self, X, y):
        raise NotImplementedError
    
    def predict_weights(self):
        raise NotImplementedError
    


