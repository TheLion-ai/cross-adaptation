"""
Simple monkey patch to fix OneHotEncoder(sparse=False) issue
"""
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder

# Store original constructor
_original_init = OneHotEncoder.__init__

def patched_init(self, *args, **kwargs):
    """Patched OneHotEncoder.__init__ that handles sparse parameter"""
    # Convert old 'sparse' parameter to new 'sparse_output' parameter
    if 'sparse' in kwargs:
        sparse_value = kwargs.pop('sparse')
        kwargs['sparse_output'] = sparse_value
    
    return _original_init(self, *args, **kwargs)

def apply_onehotencoder_patch():
    """Apply the OneHotEncoder patch"""
    print("Applying simple OneHotEncoder compatibility patch...")
    
    # Replace the __init__ method
    OneHotEncoder.__init__ = patched_init
    
    print("OneHotEncoder patch applied!")

# Auto-apply patch
apply_onehotencoder_patch()