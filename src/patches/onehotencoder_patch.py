"""
Minimal patch to fix OneHotEncoder compatibility in adapt library
"""
import sklearn.preprocessing
from sklearn.preprocessing import OneHotEncoder as OriginalOneHotEncoder

class FixedOneHotEncoder(OriginalOneHotEncoder):
    """OneHotEncoder that handles the sparse parameter correctly"""
    
    def __init__(self, *, categories='auto', drop=None, sparse=None, sparse_output=None, 
                 dtype=np.float64, handle_unknown='error', min_frequency=None, 
                 max_categories=None, feature_name_combiner='concat'):
        
        # Handle the sparse/sparse_output parameter transition
        if sparse is not None and sparse_output is None:
            sparse_output = sparse
        elif sparse_output is None:
            sparse_output = True  # Default value
            
        # Initialize with the correct parameter name for this scikit-learn version
        import inspect
        sig = inspect.signature(OriginalOneHotEncoder.__init__)
        
        if 'sparse_output' in sig.parameters:
            # New API
            super().__init__(categories=categories, drop=drop, sparse_output=sparse_output,
                           dtype=dtype, handle_unknown=handle_unknown, min_frequency=min_frequency,
                           max_categories=max_categories, feature_name_combiner=feature_name_combiner)
        else:
            # Old API  
            super().__init__(categories=categories, drop=drop, sparse=sparse_output,
                           dtype=dtype, handle_unknown=handle_unknown)

def monkey_patch_onehotencoder():
    """Replace sklearn's OneHotEncoder with our fixed version"""
    print("Applying OneHotEncoder compatibility patch...")
    
    # Import numpy here to avoid issues with the class definition
    import numpy as np
    
    # Update the class to include numpy
    FixedOneHotEncoder.__globals__['np'] = np
    
    # Replace OneHotEncoder everywhere it might be used
    sklearn.preprocessing.OneHotEncoder = FixedOneHotEncoder
    
    # Replace in sys.modules to catch any imports that already happened
    import sys
    for name, module in list(sys.modules.items()):
        if hasattr(module, 'OneHotEncoder') and module.OneHotEncoder is OriginalOneHotEncoder:
            setattr(module, 'OneHotEncoder', FixedOneHotEncoder)
    
    print("OneHotEncoder compatibility patch applied!")

# Apply patch on import
monkey_patch_onehotencoder()