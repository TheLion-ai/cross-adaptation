"""
Compatibility patch for adapt library with newer scikit-learn versions
"""
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder

# Store the original OneHotEncoder
_original_OneHotEncoder = OneHotEncoder

class CompatibleOneHotEncoder(_original_OneHotEncoder):
    """OneHotEncoder that handles both old and new scikit-learn APIs"""
    
    def __init__(self, *args, **kwargs):
        # Handle the sparse parameter renaming
        if 'sparse' in kwargs:
            sparse_value = kwargs.pop('sparse')
            # In newer scikit-learn, 'sparse' became 'sparse_output'
            try:
                # Try with sparse_output (new API)
                super().__init__(*args, sparse_output=sparse_value, **kwargs)
            except TypeError:
                # Fallback to sparse (old API)
                super().__init__(*args, sparse=sparse_value, **kwargs)
        else:
            super().__init__(*args, **kwargs)

def apply_sklearn_compatibility_patch():
    """Apply sklearn compatibility patches"""
    print("Applying scikit-learn compatibility patch...")
    
    # Replace OneHotEncoder in sklearn.preprocessing module
    sklearn.preprocessing.OneHotEncoder = CompatibleOneHotEncoder
    
    # Also replace in any modules that might have already imported it
    import sys
    for module_name, module in sys.modules.items():
        if hasattr(module, 'OneHotEncoder') and module.OneHotEncoder == _original_OneHotEncoder:
            module.OneHotEncoder = CompatibleOneHotEncoder
    
    print("Scikit-learn compatibility patch applied successfully!")

# Auto-apply patch when imported
apply_sklearn_compatibility_patch()