"""
Direct patch for TrAdaBoost OneHotEncoder compatibility issue
"""
import sys
from adapt.instance_based._tradaboost import TrAdaBoost
from sklearn.preprocessing import OneHotEncoder
import numpy as np

def patched_boost(self, iboost, X, y, sample_weight, random_state):
    """Patched _boost method that fixes OneHotEncoder compatibility"""
    
    # Original method's functionality with OneHotEncoder fix
    estimator = self._make_estimator(random_state=random_state)
    estimator.fit(X, y, sample_weight=sample_weight)
    
    y_predict = estimator.predict(X)
    
    n_classes = len(np.unique(y))
    
    if n_classes == 2:
        # Binary classification
        incorrect = y_predict != y
        estimator_error = np.average(incorrect, weights=sample_weight)
        
        if estimator_error <= 0:
            return sample_weight, 1., 0.
            
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                               'ensemble is worse than random, ensemble '
                               'can not be fitted.')
            return None, None, None
            
        alpha = self.lr * np.log((1. - estimator_error) / estimator_error)
        sample_weight *= np.exp(alpha * incorrect * (n_classes - 1) / n_classes)
        
        return sample_weight, alpha, estimator_error
        
    else:
        # Multi-class classification - this is where the OneHotEncoder issue occurs
        y_predict_proba = estimator.predict_proba(X)
        
        # Fix: Use try-except to handle both old and new OneHotEncoder APIs
        try:
            # New scikit-learn API (>=1.2)
            ohe = OneHotEncoder(sparse_output=False)
        except TypeError:
            try:
                # Old scikit-learn API (<1.2)
                ohe = OneHotEncoder(sparse=False)  
            except TypeError:
                # Very old API
                ohe = OneHotEncoder()
        
        y_coded = ohe.fit_transform(y.reshape(-1, 1))
        
        estimator_error = (-((n_classes - 1) / n_classes) * 
                          np.sum(y_coded * np.log(y_predict_proba + 1e-10), axis=1))
        estimator_error = np.average(estimator_error, weights=sample_weight)
        
        if estimator_error <= 0:
            return sample_weight, 1., 0.
            
        if estimator_error >= 1. - (1. / n_classes):
            self.estimators_.pop(-1) 
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                               'ensemble is worse than random, ensemble '
                               'can not be fitted.')
            return None, None, None
            
        alpha = (self.lr * np.log((1. - estimator_error) / estimator_error) * 
                (n_classes - 1) / n_classes)
        
        incorrect = (y_predict != y).astype(int)
        sample_weight *= np.exp(alpha * incorrect)
        
        return sample_weight, alpha, estimator_error

def apply_tradaboost_fix():
    """Apply the TrAdaBoost fix"""
    print("Applying TrAdaBoost OneHotEncoder compatibility fix...")
    
    # Replace the _boost method
    TrAdaBoost._boost = patched_boost
    
    print("TrAdaBoost fix applied successfully!")

# Auto-apply when imported
apply_tradaboost_fix()