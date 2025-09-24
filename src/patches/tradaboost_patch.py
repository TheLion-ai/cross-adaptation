"""
Patch for TrAdaBoost to fix scikit-learn compatibility issue
"""
import sys
from adapt.instance_based._tradaboost import TrAdaBoost
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Store original _boost method
_original_boost = TrAdaBoost._boost

def _patched_boost(self, iboost, X, y, sample_weight, random_state):
    """
    Patched version of _boost method that fixes OneHotEncoder compatibility
    """
    # Get the original method's code but fix the OneHotEncoder call
    n_classes = len(np.unique(y))
    
    if n_classes == 2:
        # Binary classification - use original method with fixed OneHotEncoder
        estimator = self._make_estimator(random_state=random_state)
        
        # Fit the weak learner
        estimator.fit(X, y, sample_weight=sample_weight)
        
        # Get predictions
        y_predict = estimator.predict(X)
        
        # Calculate error
        incorrect = y_predict != y
        estimator_error = np.average(incorrect, weights=sample_weight)
        
        # If perfect predictor, stop boosting
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0
        
        # If worse than random, stop boosting
        if estimator_error >= 1.0 - (1.0 / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                               'ensemble is worse than random, ensemble '
                               'can not be fitted.')
            return None, None, None
        
        # Calculate alpha
        alpha = self.lr * np.log((1.0 - estimator_error) / estimator_error)
        
        # Update sample weights
        sample_weight *= np.exp(alpha * incorrect * (n_classes - 1) / n_classes)
        
        return sample_weight, alpha, estimator_error
    
    else:
        # Multi-class classification - use patched OneHotEncoder
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)
        
        y_predict = estimator.predict(X)
        y_predict_proba = estimator.predict_proba(X)
        
        # Fix the OneHotEncoder call - use sparse_output instead of sparse
        try:
            # Try new scikit-learn API first
            ohe = OneHotEncoder(sparse_output=False)
        except TypeError:
            # Fallback to old API
            ohe = OneHotEncoder(sparse=False)
        
        y_coded = ohe.fit_transform(y.reshape(-1, 1))
        y_predict_coded = ohe.transform(y_predict.reshape(-1, 1))
        
        # Calculate error
        estimator_error = (-((n_classes - 1) / n_classes) * 
                          np.sum(y_coded * np.log(y_predict_proba + 1e-10), axis=1))
        estimator_error = np.average(estimator_error, weights=sample_weight)
        
        # If perfect predictor, stop boosting
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0
        
        # If worse than random, stop boosting  
        if estimator_error >= 1.0 - (1.0 / n_classes):
            self.estimators_.pop(-1)
            if len(self.estimators_) == 0:
                raise ValueError('BaseClassifier in AdaBoostClassifier '
                               'ensemble is worse than random, ensemble '
                               'can not be fitted.')
            return None, None, None
        
        # Calculate alpha
        alpha = self.lr * np.log((1.0 - estimator_error) / estimator_error) * (n_classes - 1) / n_classes
        
        # Update sample weights
        incorrect = (y_predict != y).astype(int)
        sample_weight *= np.exp(alpha * incorrect)
        
        return sample_weight, alpha, estimator_error

# Apply the patch
def apply_tradaboost_patch():
    """Apply the TrAdaBoost patch to fix scikit-learn compatibility"""
    print("Applying TrAdaBoost patch for scikit-learn compatibility...")
    TrAdaBoost._boost = _patched_boost
    print("TrAdaBoost patch applied successfully!")

# Auto-apply patch when module is imported
apply_tradaboost_patch()