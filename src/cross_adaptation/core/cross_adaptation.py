from typing import Callable, Dict, List, Optional
import os
from datetime import datetime

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.preprocessing import StandardScaler


class Adapter:
    def __init__(
        self,
        train_data: Dict[str, pd.DataFrame],
        adapt_model: object,
        estimator: object,
        scaler: Optional[StandardScaler] = None,
    ) -> None:
        """Cross-adaptation method for domain adaptation

        Args:
            train_data (Dict[str, pd.DataFrame]): the data to be adapted, each dataframe should have a column called target
            adapt_model (object): the method for domain adaptation
            estimator (object): estimator used for the task
            scaler (Optional[StandardScaler], optional): Scaler used for scaling the data. Defaults to None.

        Returns:
            List[pd.DataFrame]: a list of adapted dataframes
        """
        self.train_data = train_data
        self.adapt_model = adapt_model
        self.estimator = estimator
        self.scaler = scaler or self.init_scaler()

        self.__validate_input()

    def __validate_input(self):
        # Validate adapt_model
        if (
            getattr(self.adapt_model, "fit", None) is None
            or getattr(self.adapt_model, "predict_weights", None) is None
        ):
            raise ValueError("da_method must implement fit and predict_weights method")
        # Validate estimator
        if (
            getattr(self.estimator, "fit", None) is None
            or getattr(self.estimator, "predict", None) is None
        ):
            raise ValueError("estimator must implement fit and predict method")
        # validate train_data
        for dataset_name, data in self.train_data.items():
            if "target" not in data.columns:
                raise ValueError(
                    f"dataset {dataset_name} must have a column called target"
                )
        # validate scaler
        if (
            getattr(self.scaler, "fit", None) is None
            or getattr(self.scaler, "transform", None) is None
        ):
            raise ValueError("scaler must implement fit and transform method")

    def init_scaler(self) -> StandardScaler:
        """Create a scaler for the data if no scaler is provided.
        The scaler is saved in a file called scaler.bin"""
        scaler = StandardScaler()
        dfs = pd.concat(self.train_data.values(), join="outer", axis=0)
        X = dfs.loc[:, dfs.columns != "target"]
        scaler.fit(X)
        dump(scaler, "scaler.bin", compress=True)
        return scaler

    def adapt(self, save_dir: Optional[str] = None):
        """Cross-adapt the data using the adapt_model
        
        Args:
            save_dir (Optional[str]): Directory to save adapted data. If None, creates a timestamped directory.
        
        Returns:
            Dict[str, Dict]: Dictionary containing adapted data and weights for each dataset
        """
        self.adapted_data = {}
        
        # Create save directory if specified
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"adapted_data_{timestamp}"
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Adapt each dataset based on the others
        for dataset_name, dfs in self.train_data.items():
            Xs, ys = dfs.loc[:, dfs.columns != "target"], dfs["target"]
            Xs_scaled = self.scaler.transform(Xs)

            # Adapt source dataset based on the target (all other datasets)
            target = self.train_data.copy()
            del target[dataset_name] # Delete from the target the dataset used as source
            dft = pd.concat(list(target.values()), join="outer", axis=0)
            dft = dft.sample(frac=1).reset_index(drop=True)
            Xt, yt = dft.loc[:, dft.columns != "target"], dft["target"]
            Xt = self.scaler.transform(Xt)
            
            # Set target data for the adapt model (if it has these attributes)
            if hasattr(self.adapt_model, 'Xt'):
                self.adapt_model.Xt = Xt
            if hasattr(self.adapt_model, 'yt'):
                self.adapt_model.yt = np.array(yt)

            # Fit the adaptation model
            # For instance-based methods like KLIEP and KMM, we need to pass Xt as keyword argument
            # The adapt package expects fit(X, y, Xt=None, yt=None) signature
            self.adapt_model.fit(Xs_scaled, np.array(ys), Xt=Xt, yt=np.array(yt))

            # Get weights from the adaptation model
            weights = self.adapt_model.predict_weights()
            
            # Store adapted data
            Xs_df = pd.DataFrame(Xs_scaled, columns=dfs.columns.drop("target"))
            ys = ys.reset_index(drop=True)
            
            self.adapted_data[dataset_name] = {
                'X': Xs_df,
                'y': ys,
                'weights': weights,
                'X_original': Xs,  # Keep original unscaled data
            }
            
            # Save adapted data to files
            adapted_df = pd.DataFrame(Xs_scaled, columns=dfs.columns.drop("target"))
            adapted_df['target'] = ys.values
            adapted_df['weights'] = weights
            
            # Save to CSV
            file_path = os.path.join(save_dir, f"{dataset_name}_adapted.csv")
            adapted_df.to_csv(file_path, index=False)
            
            # Save weights separately as numpy array
            weights_path = os.path.join(save_dir, f"{dataset_name}_weights.npy")
            np.save(weights_path, weights)
        
        print(f"Adapted data saved to {save_dir}")
        return self.adapted_data


    def train_on_adapted_data(
        self, 
        test_data: Dict[str, pd.DataFrame], 
        metrics: List[Callable],
        use_weights: bool = True,
        save_model: bool = True
    ) -> Dict[str, float]:
        """Train the estimator on the adapted data with instance weights
        
        Args:
            test_data (Dict[str, pd.DataFrame]): Test data for evaluation
            metrics (List[Callable]): List of metrics to evaluate
            use_weights (bool): Whether to use instance weights during training
            save_model (bool): Whether to save the trained model
        
        Returns:
            Dict[str, float]: Results of the evaluation
        """
        if not hasattr(self, 'adapted_data'):
            raise ValueError("Must call adapt() method first to generate adapted data")
        
        # Combine all adapted datasets
        X_list = []
        y_list = []
        weights_list = []
        
        for dataset_name, adapted in self.adapted_data.items():
            X_list.append(adapted['X'])
            y_list.append(adapted['y'])
            weights_list.append(adapted['weights'])
        
        # Concatenate all data
        X_train = pd.concat(X_list, axis=0, ignore_index=True)
        y_train = pd.concat(y_list, axis=0, ignore_index=True)
        weights_train = np.concatenate(weights_list)
        
        # Train the estimator with or without weights
        if use_weights and hasattr(self.estimator, 'fit'):
            # Check if the estimator supports sample_weight parameter
            import inspect
            fit_params = inspect.signature(self.estimator.fit).parameters
            if 'sample_weight' in fit_params:
                self.estimator.fit(X_train, y_train, sample_weight=weights_train)
            else:
                print(f"Warning: {type(self.estimator).__name__} does not support sample_weight. Training without weights.")
                self.estimator.fit(X_train, y_train)
        else:
            self.estimator.fit(X_train, y_train)
        
        # Evaluate on test data
        results = {}
        for dataset_name, dft in test_data.items():
            Xt, yt = dft.loc[:, dft.columns != "target"], dft["target"]
            Xt = self.scaler.transform(Xt)
            y_pred = self.estimator.predict(Xt)
            
            for metric in metrics:
                metric_name = metric.func.__name__ if hasattr(metric, 'func') else metric.__name__
                results[f"adapted_{dataset_name}_{metric_name}"] = metric(yt, y_pred)
        
        # Save the trained model
        if save_model:
            dump(self.estimator, "adapted_estimator.bin", compress=True)
            print("Adapted model saved to adapted_estimator.bin")
        
        return results
    
    def calc_baseline(
        self, test_data: Dict[str, pd.DataFrame], metrics: List[Callable]
    ) -> Dict[str, float]:
        """Calculate the baseline training the model on not adapted data"""
        dfs = pd.concat(list(self.train_data.values()), join="outer", axis=0)
        Xs, ys = dfs.loc[:, dfs.columns != "target"], dfs["target"]
        Xs = self.scaler.transform(Xs)
        self.estimator.fit(Xs, np.array(ys))
        results = {}
        for dataset_name, dft in test_data.items():
            Xt, yt = dft.loc[:, dft.columns != "target"], dft["target"]
            Xt = self.scaler.transform(Xt)
            y_pred = self.estimator.predict(Xt)
            for metric in metrics:
                metric_name = metric.func.__name__ if hasattr(metric, 'func') else metric.__name__
                results[f"baseline_{dataset_name}_{metric_name}"] = metric(yt, y_pred)
        dump(self.estimator, "baseline_estimator.bin", compress=True)
        return results

    def load_adapted_data(self, save_dir: str) -> Dict[str, Dict]:
        """Load previously saved adapted data from directory
        
        Args:
            save_dir (str): Directory containing saved adapted data
        
        Returns:
            Dict[str, Dict]: Dictionary containing adapted data and weights for each dataset
        """
        self.adapted_data = {}
        
        # List all CSV files in the directory
        csv_files = [f for f in os.listdir(save_dir) if f.endswith('_adapted.csv')]
        
        for csv_file in csv_files:
            dataset_name = csv_file.replace('_adapted.csv', '')
            
            # Load adapted data
            file_path = os.path.join(save_dir, csv_file)
            adapted_df = pd.read_csv(file_path)
            
            # Separate features, target, and weights
            X = adapted_df.drop(['target', 'weights'], axis=1)
            y = adapted_df['target']
            weights = adapted_df['weights'].values
            
            # Also try to load weights from numpy file if available
            weights_path = os.path.join(save_dir, f"{dataset_name}_weights.npy")
            if os.path.exists(weights_path):
                weights = np.load(weights_path)
            
            self.adapted_data[dataset_name] = {
                'X': X,
                'y': y,
                'weights': weights
            }
        
        print(f"Loaded adapted data from {save_dir}")
        return self.adapted_data

    

    def compare(
        self,
        results: Dict[str, float],
        baseline_results: Dict[str, float],
        metrics: List[object],
        test_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """Compare the results of the baseline and the adapted model

        Args:
            baseline_results (Dict[str, float]): the results of the baseline model
            metrics (List[Callable], optional): the metrics to be used for evaluation. Each metric should ba a callable that accepts 2 array like objects of true and pred values ad return a float.
            test_data (Dict[str, pd.DataFrame]): the test data
        return:
            Dict[str, float]: dictionary of metrics and values
        """

        compared_results = {}
        for metric in metrics:
            for dataset_name in test_data.keys():
                metric_name = metric.func.__name__ if hasattr(metric, 'func') else metric.__name__
                key = f"{dataset_name}_{metric_name}"
                results_metric = list(
                    filter(lambda item: key in item[0], results.items())
                )[0][1]
                baseline_results_metric = list(
                    filter(lambda item: key in item[0], baseline_results.items())
                )[0][1]
                compared_results[f"compared_{key}"] = (
                    results_metric - baseline_results_metric
                )
        return compared_results
