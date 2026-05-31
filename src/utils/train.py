import numpy as np
import torch
from torch import nn
from transformers import Trainer
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

class WeightedLossTrainer(Trainer):
    """
    Custom Trainer that applies class weights to the loss function.
    """
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The trainer automatically moves things to the correct device via self.args.device
        self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    

def _visualize_smoothing_effect(balanced_weights: np.ndarray, smoothed_weights: np.ndarray):
    """
    Visualizes the effect of weight smoothing using log1p transformation.
    Used in: check_weight_strategy() when visualize=True.
    
    Args:
        balanced_weights (np.ndarray): Balanced class weights from sklearn.
        smoothed_weights (np.ndarray): Smoothed class weights after transformation.
    """
    
    # Weights check (before vs after smoothing)
    print(f"Unsmoothed class weights range: {balanced_weights.min():.2f} → {balanced_weights.max():.2f}")
    print(f"Smoothed class weights range: {smoothed_weights.min():.2f} → {smoothed_weights.max():.2f}")
    
    plt.figure(figsize=(12, 6))
    plt.plot(sorted(balanced_weights), label='Original')
    plt.plot(sorted(smoothed_weights), label='Smoothed')
    plt.legend()
    plt.title("Class Weight Smoothing Effect")
    plt.xlabel("Class Index (sorted)")
    plt.ylabel("Weight")
    plt.show()
    
def check_weight_strategy(
    labels: list = [], 
    smoothing_factor: float = 1.0,
    threshold: float = 10.0,
    visualize: bool = False
):
    """
    Analyzes class distribution and recommends a weighting strategy.
    
    Args
    ----------
        labels (list): Training targets/labels.
        threshold (float): Imbalance ratio cutoff (default is 10:1).
        visualize (bool): Whether to visualize the smoothing effect (default is False).
        smoothing_factor (float): Controls the degree of smoothing (default is 1.0).

    Returns
    -------
        str: Weight strategy recommendation ('balanced', 'smoothed', or 'none').

    Examples:
    ----------
        labels = hf_dataset['train']['label']
        check_weight_strategy(labels, visualize=True)

        # Example 1: Severe imbalance
        print("Test Case A:")
        check_weight_strategy([0]*1000 + [1]*5, visualize=True) 

        # Example 2: Mild imbalance
        print("Test Case B:")
        check_weight_strategy([0]*100 + [1]*30)
    """
    classes, counts = np.unique(labels, return_counts=True)
    
    max_count = np.max(counts)
    min_count = np.min(counts)
    
    # Calculate how many times more frequent the majority class is
    imbalance_ratio = max_count / min_count
    
    print("--- IMBALANCE ANALYSIS ---")
    print(f"Majority class count: {max_count}")
    print(f"Minority class count: {min_count}")
    print(f"Imbalance ratio:      {imbalance_ratio:.2f}:1\n")
    
    # Compute balanced weights for visualization/reference
    balanced_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels
    )

    # Normalize weights to have mean=1 before smoothing
    normalized_weights = balanced_weights / balanced_weights.mean()
    
    # Apply log1p transformation to compress extreme values
    smoothed_weights = np.log1p(normalized_weights * smoothing_factor) + 1.0
    
    print("--- RECOMMENDATION ---")
    if imbalance_ratio >= threshold:
        print("Use SMOOTHED weights.")
        print(f"Reason: Imbalance exceeds {threshold}:1.")
        print("Raw weights will destabilize BERT gradients.")
        if visualize:
            _visualize_smoothing_effect(balanced_weights, smoothed_weights)
        return "smoothed", smoothed_weights
    elif imbalance_ratio > 1.05:  # >5% variance
        print("Use NORMAL (balanced) weights.")
        print(f"Reason: Imbalance is mild (under {threshold}:1).")
        print("Model will safely handle raw scaling.")
        if visualize:
            balanced_weights = compute_class_weight(
                class_weight="balanced",
                classes=classes,
                y=labels
            )
            _visualize_smoothing_effect(balanced_weights, smoothed_weights)
        return "balanced", balanced_weights
    else:
        print("Use NO weights (Uniform).")
        print("Reason: Dataset is perfectly balanced.")
        return "none", None