from typing import Callable, Dict, List, Optional, Tuple
import os
from datetime import datetime
import inspect

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
        """
        self.train_data = train_data
        self.adapt_model = adapt_model
        self.estimator = estimator
        self.scaler = scaler or self._create_scaler()
        self.adapted_data = {}

        self._validate_inputs()

    def _validate_inputs(self):
        """Validate all inputs to ensure they have required methods and attributes"""
        self._validate_model(self.adapt_model, ["fit", "predict_weights"], "adapt_model")
        self._validate_model(self.estimator, ["fit", "predict"], "estimator")
        self._validate_model(self.scaler, ["fit", "transform"], "scaler")
        self._validate_train_data()

    def _validate_model(self, model: object, required_methods: List[str], model_name: str):
        """Validate that a model has all required methods"""
        for method in required_methods:
            if not hasattr(model, method) or not callable(getattr(model, method)):
                raise ValueError(f"{model_name} must implement {method} method")

    def _validate_train_data(self):
        """Validate that all datasets have a target column"""
        for dataset_name, data in self.train_data.items():
            if "target" not in data.columns:
                raise ValueError(f"dataset {dataset_name} must have a column called target")

    def _create_scaler(self) -> StandardScaler:
        """Create and fit a scaler on all training data"""
        scaler = StandardScaler()
        all_data = pd.concat(self.train_data.values(), join="outer", axis=0)
        features = all_data.loc[:, all_data.columns != "target"]
        scaler.fit(features)
        dump(scaler, "scaler.bin", compress=True)
        return scaler

    def adapt(self, save_dir: Optional[str] = None) -> Dict[str, Dict]:
        """Cross-adapt the data using the adapt_model

        Args:
            save_dir (Optional[str]): Directory to save adapted data. If None, creates a timestamped directory.

        Returns:
            Dict[str, Dict]: Dictionary containing adapted data and weights for each dataset
        """
        save_dir = self._prepare_save_directory(save_dir)
        self.adapted_data = {}

        for dataset_name, source_data in self.train_data.items():
            print(f"Processing dataset: {dataset_name}")

            # Prepare source and target data
            source_features, source_targets = self._extract_features_and_targets(source_data)
            source_features_scaled = self.scaler.transform(source_features)

            target_data, target_weights = self._prepare_target_data(dataset_name)
            target_features_scaled = self.scaler.transform(target_data.drop('target', axis=1))
            target_targets = target_data['target']

            # Fit adaptation model and get weights
            weights = self._fit_and_get_weights(
                source_features_scaled, source_targets,
                target_features_scaled, target_targets,
                target_weights, dataset_name
            )

            # Store and save adapted data
            self._store_adapted_data(dataset_name, source_features_scaled, source_targets,
                                   source_features, weights)
            self._save_adapted_data(dataset_name, source_features_scaled, source_targets,
                                  weights, save_dir)

        print(f"Adapted data saved to {save_dir}")
        return self.adapted_data

    def _prepare_save_directory(self, save_dir: Optional[str]) -> str:
        """Prepare the directory for saving adapted data"""
        if save_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = f"adapted_data_{timestamp}"

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        return save_dir

    def _extract_features_and_targets(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Extract features and targets from a dataset"""
        features = data.loc[:, data.columns != "target"]
        targets = data["target"]
        return features, targets

    def _prepare_target_data(self, exclude_dataset: str) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare target data by combining all datasets except the excluded one"""
        target_datasets = {k: v for k, v in self.train_data.items() if k != exclude_dataset}

        target_dataframes = []
        weights = []

        for dataset_name, dataset_df in target_datasets.items():
            target_dataframes.append(dataset_df)
            # Weight inversely proportional to dataset size for equal contribution
            dataset_weight = 1.0 / len(dataset_df)
            weights.extend([dataset_weight] * len(dataset_df))

        combined_target = pd.concat(target_dataframes, join="outer", axis=0).reset_index(drop=True)

        # Normalize weights and shuffle data
        weights = np.array(weights)
        num_subdatasets = len(target_datasets)
        weights = weights * num_subdatasets / weights.sum()

        # Shuffle data and weights together
        shuffle_indices = np.random.RandomState(42).permutation(len(combined_target))
        combined_target = combined_target.iloc[shuffle_indices].reset_index(drop=True)
        weights = weights[shuffle_indices]

        return combined_target, weights

    def _fit_and_get_weights(self, source_features: np.ndarray, source_targets: pd.Series,
                           target_features: np.ndarray, target_targets: pd.Series,
                           target_weights: np.ndarray, dataset_name: str) -> np.ndarray:
        """Fit the adaptation model and get instance weights"""
        # Set target data attributes if model supports them
        if hasattr(self.adapt_model, 'Xt'):
            self.adapt_model.Xt = target_features
        if hasattr(self.adapt_model, 'yt'):
            self.adapt_model.yt = np.array(target_targets)

        # Try to fit the model with different parameter signatures
        try:
            self._fit_adaptation_model(source_features, source_targets,
                                     target_features, target_targets, target_weights)
        except (ValueError, ArithmeticError) as e:
            print(f"Warning: Adaptation method failed for {dataset_name} with error: {e}")
            return self._create_fallback_weights(source_features, dataset_name)

        # Get weights from the model
        return self._get_weights_from_model(source_features, dataset_name)

    def _fit_adaptation_model(self, source_features: np.ndarray, source_targets: pd.Series,
                            target_features: np.ndarray, target_targets: pd.Series,
                            target_weights: np.ndarray):
        """Fit the adaptation model with appropriate parameters"""
        fit_params = inspect.signature(self.adapt_model.fit).parameters

        if 'sample_weight_target' in fit_params:
            self.adapt_model.fit(source_features, np.array(source_targets),
                               Xt=target_features, yt=np.array(target_targets),
                               sample_weight_target=target_weights)
        elif 'target_weight' in fit_params:
            self.adapt_model.fit(source_features, np.array(source_targets),
                               Xt=target_features, yt=np.array(target_targets),
                               target_weight=target_weights)
        else:
            if hasattr(self.adapt_model, 'target_weights'):
                self.adapt_model.target_weights = target_weights
            self.adapt_model.fit(source_features, np.array(source_targets),
                               Xt=target_features, yt=np.array(target_targets))

    def _create_fallback_weights(self, source_features: np.ndarray, dataset_name: str) -> np.ndarray:
        """Create uniform fallback weights when adaptation fails"""
        print("Falling back to uniform weights...")
        uniform_weights = np.ones(len(source_features))

        # Store fallback information
        if not hasattr(self.adapt_model, '_fallback_datasets'):
            self.adapt_model._fallback_datasets = set()
        self.adapt_model._fallback_datasets.add(dataset_name)
        self.adapt_model._fallback_weights = uniform_weights

        print(f"Using uniform weights for {dataset_name}")
        return uniform_weights

    def _get_weights_from_model(self, source_features: np.ndarray, dataset_name: str) -> np.ndarray:
        """Get weights from the adaptation model with fallback handling"""
        # Check if using fallback weights
        if (hasattr(self.adapt_model, '_fallback_datasets') and
            dataset_name in self.adapt_model._fallback_datasets):
            print(f"Using fallback uniform weights for {dataset_name}")
            return self.adapt_model._fallback_weights

        # Try to get weights from the model
        try:
            weights = self.adapt_model.predict_weights()
        except (ValueError, ArithmeticError) as e:
            print(f"Warning: predict_weights failed for {dataset_name} with error: {e}")
            print("Using uniform weights as fallback...")
            weights = np.ones(len(source_features))

        return self._validate_and_normalize_weights(weights, dataset_name)

    def _validate_and_normalize_weights(self, weights: np.ndarray, dataset_name: str) -> np.ndarray:
        """Validate and normalize instance weights"""
        # Fix non-positive weights
        if np.any(weights <= 0):
            print(f"Warning: Found {np.sum(weights <= 0)} non-positive weights in {dataset_name}. Fixing...")
            min_positive_weight = weights[weights > 0].min() if np.any(weights > 0) else 1e-6
            weights[weights <= 0] = min_positive_weight * 0.01
            print(f"Replaced non-positive weights with {min_positive_weight * 0.01:.2e}")

        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        print(f"Dataset {dataset_name}: weights range [{weights.min():.6f}, {weights.max():.6f}], sum: {weights.sum():.1f}")

        return weights

    def _store_adapted_data(self, dataset_name: str, features_scaled: np.ndarray,
                          targets: pd.Series, features_original: pd.DataFrame, weights: np.ndarray):
        """Store adapted data in memory"""
        features_df = pd.DataFrame(features_scaled, columns=features_original.columns)
        targets_reset = targets.reset_index(drop=True)

        self.adapted_data[dataset_name] = {
            'X': features_df,
            'y': targets_reset,
            'weights': weights,
            'X_original': features_original,
        }

    def _save_adapted_data(self, dataset_name: str, features_scaled: np.ndarray,
                         targets: pd.Series, weights: np.ndarray, save_dir: str):
        """Save adapted data to files"""
        # Create dataframe with scaled features, targets, and weights
        feature_columns = [col for col in self.train_data[dataset_name].columns if col != 'target']
        adapted_df = pd.DataFrame(features_scaled, columns=feature_columns)
        adapted_df['target'] = targets.values
        adapted_df['weights'] = weights

        # Save to CSV
        csv_path = os.path.join(save_dir, f"{dataset_name}_adapted.csv")
        adapted_df.to_csv(csv_path, index=False)

        # Save weights separately as numpy array
        weights_path = os.path.join(save_dir, f"{dataset_name}_weights.npy")
        np.save(weights_path, weights)


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
        if not self.adapted_data:
            raise ValueError("Must call adapt() method first to generate adapted data")

        # Combine and prepare training data
        train_features, train_targets, train_weights = self._combine_adapted_data()

        # Validate and process weights
        if use_weights:
            train_weights, use_weights = self._process_training_weights(train_weights)

        # Train the estimator
        self._train_estimator(train_features, train_targets, train_weights, use_weights)

        # Evaluate and return results
        results = self._evaluate_on_test_data(test_data, metrics)

        if save_model:
            dump(self.estimator, "adapted_estimator.bin", compress=True)
            print("Adapted model saved to adapted_estimator.bin")

        return results

    def _combine_adapted_data(self) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Combine all adapted datasets into training data"""
        features_list = []
        targets_list = []
        weights_list = []

        for dataset_name, adapted in self.adapted_data.items():
            features_list.append(adapted['X'])
            targets_list.append(adapted['y'])
            weights_list.append(adapted['weights'])

        combined_features = pd.concat(features_list, axis=0, ignore_index=True)
        combined_targets = pd.concat(targets_list, axis=0, ignore_index=True)
        combined_weights = np.concatenate(weights_list)

        return combined_features, combined_targets, combined_weights

    def _process_training_weights(self, weights: np.ndarray) -> Tuple[np.ndarray, bool]:
        """Process and validate training weights"""
        print(f"Original weights - min: {weights.min():.6f}, max: {weights.max():.6f}, mean: {weights.mean():.6f}")

        # Check for invalid weights
        if np.all(weights <= 0):
            print("Warning: All weights are zero or negative. Disabling weights and using uniform weighting.")
            return weights, False
        elif np.any(weights <= 0):
            print(f"Warning: {np.sum(weights <= 0)} weights are zero or negative. Setting them to minimum positive weight.")
            min_positive_weight = weights[weights > 0].min()
            weights[weights <= 0] = min_positive_weight * 0.01

        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        print(f"Normalized weights - min: {weights.min():.6f}, max: {weights.max():.6f}, sum: {weights.sum():.1f}")

        return weights, True

    def _train_estimator(self, features: pd.DataFrame, targets: pd.Series,
                        weights: np.ndarray, use_weights: bool):
        """Train the estimator with or without sample weights"""
        if use_weights and self._estimator_supports_weights():
            self.estimator.fit(features, targets, sample_weight=weights)
        else:
            if use_weights:
                print(f"Warning: {type(self.estimator).__name__} does not support sample_weight. Training without weights.")
            self.estimator.fit(features, targets)

    def _estimator_supports_weights(self) -> bool:
        """Check if the estimator supports sample_weight parameter"""
        fit_params = inspect.signature(self.estimator.fit).parameters
        return 'sample_weight' in fit_params

    def _evaluate_on_test_data(self, test_data: Dict[str, pd.DataFrame],
                             metrics: List[Callable]) -> Dict[str, float]:
        """Evaluate the trained estimator on test data"""
        results = {}

        for dataset_name, test_df in test_data.items():
            test_features, test_targets = self._extract_features_and_targets(test_df)
            test_features_scaled = self.scaler.transform(test_features)
            predictions = self.estimator.predict(test_features_scaled)

            for metric in metrics:
                metric_name = self._get_metric_name(metric)
                results[f"adapted_{dataset_name}_{metric_name}"] = metric(test_targets, predictions)

        return results

    def _get_metric_name(self, metric: Callable) -> str:
        """Get the name of a metric function"""
        return metric.func.__name__ if hasattr(metric, 'func') else metric.__name__
    
    def calc_baseline(
        self, test_data: Dict[str, pd.DataFrame], metrics: List[Callable]
    ) -> Dict[str, float]:
        """Calculate baseline results by training on non-adapted data"""
        # Combine all training data without adaptation
        all_train_data = pd.concat(self.train_data.values(), join="outer", axis=0)
        train_features, train_targets = self._extract_features_and_targets(all_train_data)
        train_features_scaled = self.scaler.transform(train_features)

        # Train estimator on combined data
        self.estimator.fit(train_features_scaled, train_targets)

        # Evaluate on test data
        results = {}
        for dataset_name, test_df in test_data.items():
            test_features, test_targets = self._extract_features_and_targets(test_df)
            test_features_scaled = self.scaler.transform(test_features)
            predictions = self.estimator.predict(test_features_scaled)

            for metric in metrics:
                metric_name = self._get_metric_name(metric)
                results[f"baseline_{dataset_name}_{metric_name}"] = metric(test_targets, predictions)

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
        csv_files = [f for f in os.listdir(save_dir) if f.endswith('_adapted.csv')]

        for csv_file in csv_files:
            dataset_name = csv_file.replace('_adapted.csv', '')
            self.adapted_data[dataset_name] = self._load_single_adapted_dataset(save_dir, dataset_name)

        print(f"Loaded adapted data from {save_dir}")
        return self.adapted_data

    def _load_single_adapted_dataset(self, save_dir: str, dataset_name: str) -> Dict:
        """Load a single adapted dataset from files"""
        # Load CSV file
        csv_path = os.path.join(save_dir, f"{dataset_name}_adapted.csv")
        adapted_df = pd.read_csv(csv_path)

        # Separate components
        features = adapted_df.drop(['target', 'weights'], axis=1)
        targets = adapted_df['target']
        weights = adapted_df['weights'].values

        # Try to load weights from numpy file (more precise)
        weights_path = os.path.join(save_dir, f"{dataset_name}_weights.npy")
        if os.path.exists(weights_path):
            weights = np.load(weights_path)

        return {
            'X': features,
            'y': targets,
            'weights': weights
        }

    

    def compare(
        self,
        adapted_results: Dict[str, float],
        baseline_results: Dict[str, float],
        metrics: List[Callable],
        test_data: Dict[str, pd.DataFrame],
    ) -> Dict[str, float]:
        """Compare results between adapted and baseline models

        Args:
            adapted_results (Dict[str, float]): Results from the adapted model
            baseline_results (Dict[str, float]): Results from the baseline model
            metrics (List[Callable]): Metrics used for evaluation
            test_data (Dict[str, pd.DataFrame]): Test datasets

        Returns:
            Dict[str, float]: Comparison results (adapted - baseline)
        """
        comparison_results = {}

        for metric in metrics:
            metric_name = self._get_metric_name(metric)
            for dataset_name in test_data.keys():
                # Find corresponding results for this dataset and metric
                adapted_score = self._find_result_by_key(adapted_results, dataset_name, metric_name)
                baseline_score = self._find_result_by_key(baseline_results, dataset_name, metric_name)

                comparison_key = f"compared_{dataset_name}_{metric_name}"
                comparison_results[comparison_key] = adapted_score - baseline_score

        return comparison_results

    def _find_result_by_key(self, results: Dict[str, float], dataset_name: str, metric_name: str) -> float:
        """Find a specific result by dataset and metric name"""
        key_pattern = f"{dataset_name}_{metric_name}"
        matching_results = [value for key, value in results.items() if key_pattern in key]

        if not matching_results:
            raise ValueError(f"No results found for dataset '{dataset_name}' and metric '{metric_name}'")

        return matching_results[0]
