"""XGBoost Federated Learning - Client Module"""

import os
from typing import Dict, Tuple, List
import flwr as fl
from flwr.common import NDArrays, Scalar
import xgboost as xgb
import numpy as np
from pathlib import Path


class XGBoostClient(fl.client.NumPyClient):
    """Flower client for federated XGBoost training.
    
    Supports two training methods:
    - bagging: Each client trains new trees, server combines all trees
    - cyclic: Each client refines the global model sequentially
    """
    
    def __init__(
        self,
        local_data: Dict,
        client_id: str = "client",
        saving_path: str = "/sandbox/",
    ):
        """
        Initialize XGBoost client.
        
        Args:
            local_data: Dictionary containing:
                - X_train: Training features
                - y_train: Training labels
                - X_test: Test features
                - y_test: Test labels
            client_id: Unique identifier for this client
            saving_path: Path to save local models and logs
        """
        self.local_data = local_data
        self.client_id = client_id
        self.saving_path = Path(saving_path)
        self.saving_path.mkdir(parents=True, exist_ok=True)
        
        # Create models directory
        models_dir = self.saving_path / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Local model
        self.bst = None
        self.xgb_params = {}
        self.dtrain = None
        self.dtest = None
        
        # Prepare data
        self._prepare_data()
        
        print(f"[Client {self.client_id}] Initialized")
        print(f"[Client {self.client_id}] Training samples: {len(self.local_data['X_train'])}")
        print(f"[Client {self.client_id}] Test samples: {len(self.local_data['X_test'])}")
    
    def _prepare_data(self):
        """Convert data to DMatrix format for XGBoost."""
        X_train = self.local_data['X_train']
        y_train = self.local_data['y_train']
        X_test = self.local_data['X_test']
        y_test = self.local_data['y_test']
        
        # Create DMatrix objects
        self.dtrain = xgb.DMatrix(X_train, label=y_train)
        self.dtest = xgb.DMatrix(X_test, label=y_test)
        
        print(f"[Client {self.client_id}] Data prepared as DMatrix")
    
    def get_parameters(self, config: Dict[str, Scalar] = None) -> NDArrays:
        """Return current model parameters."""
        if self.bst is None:
            # Return empty parameters if no model yet
            return [np.array([], dtype=np.uint8)]
        
        # Serialize model
        model_bytes = self.bst.save_raw("json")
        return [np.frombuffer(model_bytes, dtype=np.uint8)]
    
    def set_parameters(self, parameters: NDArrays):
        """Set model parameters from server."""
        if len(parameters) == 0 or len(parameters[0]) == 0:
            # No parameters to load (first round)
            self.bst = None
            return
        
        # Load model from bytes
        model_bytes = bytearray(parameters[0].tobytes())
        self.bst = xgb.Booster(params=self.xgb_params)
        self.bst.load_model(model_bytes)
        
        print(f"[Client {self.client_id}] Loaded global model with {self.bst.num_boosted_rounds()} trees")
    
    def fit(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        """Train the model on local data.
        
        Args:
            parameters: Model parameters from server
            config: Training configuration from server
        
        Returns:
            Tuple of (updated_parameters, num_examples, metrics)
        """
        
        # Extract config
        server_round = int(config.get("server_round", 1))
        num_local_rounds = int(config.get("num_local_rounds", 5))
        train_method = config.get("train_method", "bagging")
        
        # Update XGBoost parameters from config
        self.xgb_params = {
            k: v for k, v in config.items()
            if k not in ["server_round", "num_local_rounds", "train_method"]
        }
        
        print(f"\n[Client {self.client_id}] === Round {server_round} - FIT ===")
        print(f"[Client {self.client_id}] Method: {train_method}")
        print(f"[Client {self.client_id}] Local rounds: {num_local_rounds}")
        
        if server_round == 1:
            # First round: train from scratch
            print(f"[Client {self.client_id}] Training from scratch...")
            self.bst = xgb.train(
                self.xgb_params,
                self.dtrain,
                num_boost_round=num_local_rounds,
            )
        else:
            # Subsequent rounds: load global model and continue training
            self.set_parameters(parameters)
            
            if self.bst is None:
                # Fallback: train from scratch if loading failed
                print(f"[Client {self.client_id}] Warning: Could not load model, training from scratch")
                self.bst = xgb.train(
                    self.xgb_params,
                    self.dtrain,
                    num_boost_round=num_local_rounds,
                )
            else:
                # Continue training
                print(f"[Client {self.client_id}] Continuing training from global model...")
                initial_trees = self.bst.num_boosted_rounds()
                
                # Update trees based on local training data
                for i in range(num_local_rounds):
                    self.bst.update(self.dtrain, self.bst.num_boosted_rounds())
                
                final_trees = self.bst.num_boosted_rounds()
                print(f"[Client {self.client_id}] Trained {final_trees - initial_trees} new trees (total: {final_trees})")
        
        print(f"[Client {self.client_id}] Total trees in model: {self.bst.num_boosted_rounds()}")
        
        # For bagging: return only the last N trees
        # For cyclic: return the entire model
        if train_method == "bagging":
            # Extract only the newly trained trees
            num_trees = self.bst.num_boosted_rounds()
            if num_trees > num_local_rounds:
                # Slice to get last num_local_rounds trees
                model_to_send = self.bst[num_trees - num_local_rounds : num_trees]
                print(f"[Client {self.client_id}] Sending last {num_local_rounds} trees (bagging mode)")
            else:
                model_to_send = self.bst
                print(f"[Client {self.client_id}] Sending all {num_trees} trees")
        else:
            # Cyclic: send entire model
            model_to_send = self.bst
            print(f"[Client {self.client_id}] Sending entire model (cyclic mode)")
        
        # Serialize model
        model_bytes = model_to_send.save_raw("json")
        model_array = np.frombuffer(model_bytes, dtype=np.uint8)
        
        # Get number of training examples
        num_examples = len(self.local_data['X_train'])
        
        # Prepare metrics
        metrics = {
            "num_examples": num_examples,
            "num_trees": self.bst.num_boosted_rounds(),
        }
        
        # Save local model
        local_model_path = self.saving_path / "models" / f"xgboost_client_{self.client_id}_round_{server_round}.json"
        self.bst.save_model(str(local_model_path))
        print(f"[Client {self.client_id}] Saved local model to {local_model_path}")
        
        return [model_array], num_examples, metrics
    
    def evaluate(
        self,
        parameters: NDArrays,
        config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        """Evaluate the global model on local test data.
        
        Args:
            parameters: Model parameters from server
            config: Evaluation configuration from server
        
        Returns:
            Tuple of (loss, num_examples, metrics)
        """
        
        server_round = int(config.get("server_round", 0))
        
        print(f"\n[Client {self.client_id}] === Round {server_round} - EVALUATE ===")
        
        # Update XGBoost parameters
        self.xgb_params = {
            k: v for k, v in config.items()
            if k not in ["server_round"]
        }
        
        # Load global model
        self.set_parameters(parameters)
        
        if self.bst is None:
            print(f"[Client {self.client_id}] Warning: No model to evaluate")
            return 0.0, 0, {}
        
        # Evaluate on test set
        eval_results = self.bst.eval_set(
            evals=[(self.dtest, "test")],
            iteration=self.bst.num_boosted_rounds() - 1,
        )
        
        print(f"[Client {self.client_id}] Evaluation results: {eval_results}")
        
        # Parse evaluation results
        # Format: "[0]\ttest-auc:0.85123"
        metrics = {}
        try:
            parts = eval_results.split("\t")
            for part in parts[1:]:  # Skip the iteration number
                metric_name, metric_value = part.split(":")
                metric_name = metric_name.replace("test-", "")
                metrics[metric_name] = float(metric_value)
        except Exception as e:
            print(f"[Client {self.client_id}] Warning: Could not parse metrics: {e}")
        
        # Add client ID to metrics
        # Note: We don't include client_id in metrics as it's a string
        # and would cause issues during aggregation
        # metrics['client_id'] = self.client_id
        
        # Get predictions for additional metrics
        y_pred = self.bst.predict(self.dtest)
        y_true = self.local_data['y_test']
        
        # Calculate additional metrics
        if self.xgb_params.get("objective", "").startswith("binary"):
            # Binary classification
            from sklearn.metrics import accuracy_score, precision_score, recall_score
            
            y_pred_binary = (y_pred > 0.5).astype(int)
            metrics['accuracy'] = float(accuracy_score(y_true, y_pred_binary))
            metrics['precision'] = float(precision_score(y_true, y_pred_binary, zero_division=0))
            metrics['recall'] = float(recall_score(y_true, y_pred_binary, zero_division=0))
        
        # Loss is 1 - primary metric (e.g., 1 - AUC)
        primary_metric = metrics.get('auc', metrics.get('mlogloss', 0))
        loss = 1 - primary_metric if 'auc' in metrics else primary_metric
        
        num_examples = len(self.local_data['X_test'])
        
        print(f"[Client {self.client_id}] Metrics: {metrics}")
        print(f"[Client {self.client_id}] Loss: {loss:.4f}")
        
        return loss, num_examples, metrics


def get_numpy(X_train, y_train, X_test, y_test, time_col=None, event_col=None) -> Dict:
    """Convert data to dictionary format expected by client.
    
    Args:
        X_train: Training features (numpy array or pandas DataFrame)
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        time_col: Optional time column for survival analysis
        event_col: Optional event column for survival analysis
    
    Returns:
        Dictionary with X_train, y_train, X_test, y_test
    """
    
    # Convert to numpy if needed
    if hasattr(X_train, 'values'):  # pandas DataFrame
        X_train = X_train.values
    if hasattr(y_train, 'values'):  # pandas Series
        y_train = y_train.values
    if hasattr(X_test, 'values'):
        X_test = X_test.values
    if hasattr(y_test, 'values'):
        y_test = y_test.values
    
    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test,
        'num_examples': len(X_train),
    }


def get_client(config: Dict, data: Tuple, client_id: str) -> fl.client.Client:
    """Create and return XGBoost federated learning client.
    
    Args:
        config: Configuration dictionary containing experiment settings
        data: Tuple of ((X_train, y_train), (X_test, y_test), time_col, event_col)
        client_id: Unique identifier for this client
    
    Returns:
        Initialized XGBoostClient
    """
    
    (X_train, y_train), (X_test, y_test), time_col, event_col = data
    
    # Convert to format expected by client
    local_data = get_numpy(X_train, y_train, X_test, y_test, time_col, event_col)
    
    # Create client
    client = XGBoostClient(
        local_data=local_data,
        client_id=str(client_id),
        saving_path=config.get("experiment_dir", "/sandbox/"),
    )
    
    return client