import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris  # ONLY used for loading data

###############################################################################
# 1. LOAD AND SPLIT THE DATA
###############################################################################
# We will do a manual 60/40 split for training and testing.

def load_and_split_data(test_ratio=0.4, random_seed=42):
    np.random.seed(random_seed)
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data        # shape: (150, 4)
    y = iris.target      # values: 0, 1, 2 (3 classes)

    # Shuffle indices
    indices = np.random.permutation(len(X))
    train_size = int((1 - test_ratio) * len(X))
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    # Split data
    X_train = X[train_idx]
    y_train = y[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train, y_train, X_test, y_test


###############################################################################
# 2. PREPROCESSING FUNCTIONS
###############################################################################
# We implement three approaches: 
#   1) Unnormalized  (no change)
#   2) Standardized  (z-score)
#   3) Min-Max normalization (scale to [0,1])

def standardize(X, mean, std):
    return (X - mean) / (std + 1e-15)

def min_max_normalize(X, min_val, max_val):
    return (X - min_val) / (max_val - min_val + 1e-15)

def get_preprocessed_data(X_train, X_test, method='unnormalized'):
    """
    method can be:
      - 'unnormalized': returns X_train, X_test unchanged
      - 'standard': returns standardized data
      - 'minmax': returns min-max normalized data
    """
    if method == 'unnormalized':
        return X_train, X_test
    
    elif method == 'standard':
        mean = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        return standardize(X_train, mean, std), standardize(X_test, mean, std)
    
    elif method == 'minmax':
        min_val = np.min(X_train, axis=0)
        max_val = np.max(X_train, axis=0)
        return min_max_normalize(X_train, min_val, max_val), min_max_normalize(X_test, min_val, max_val)
    
    else:
        raise ValueError("Unknown method for preprocessing.")


###############################################################################
# 3. ONE-HOT ENCODING FOR CROSS-ENTROPY
###############################################################################
def one_hot_encode(y, num_classes):
    """
    y: array of shape (n_samples,) with class labels 0..(num_classes-1)
    returns: array of shape (n_samples, num_classes)
    """
    return np.eye(num_classes)[y]


###############################################################################
# 4. SOFTMAX AND CROSS-ENTROPY LOSS
###############################################################################
def softmax(z):
    """
    z: (n_samples, n_classes)
    returns: softmax probabilities (n_samples, n_classes)
    """
    # Subtract max for numerical stability
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def cross_entropy_loss(y_true, y_pred):
    """
    y_true: one-hot encoded true labels (n_samples, n_classes)
    y_pred: predicted probabilities from softmax (n_samples, n_classes)
    returns: scalar cross-entropy loss
    """
    m = y_true.shape[0]
    eps = 1e-15
    log_likelihood = -np.log(y_pred + eps)
    loss = np.sum(y_true * log_likelihood) / m
    return loss


###############################################################################
# 5. MULTI-CLASS LOGISTIC REGRESSION: GRADIENT DESCENT
###############################################################################
def train_logreg_gd(X_train, y_train_onehot, lr=0.1, epochs=100):
    """
    X_train: (n_samples, n_features)
    y_train_onehot: (n_samples, n_classes)
    lr: learning rate
    epochs: number of iterations for gradient descent
    
    returns: (W, b, loss_history)
    """
    n_samples, n_features = X_train.shape
    n_classes = y_train_onehot.shape[1]
    
    # Initialize weights and bias
    W = np.random.randn(n_features, n_classes) * 0.01
    b = np.zeros((1, n_classes))
    
    loss_history = []
    
    for epoch in range(epochs):
        # Forward pass
        logits = X_train.dot(W) + b  # shape: (n_samples, n_classes)
        probs = softmax(logits)      # shape: (n_samples, n_classes)
        
        # Compute loss
        loss = cross_entropy_loss(y_train_onehot, probs)
        loss_history.append(loss)
        
        # Backpropagation
        # dL/dlogits = (probs - y_true) / n_samples
        grad_logits = (probs - y_train_onehot) / n_samples
        grad_W = X_train.T.dot(grad_logits)   # shape: (n_features, n_classes)
        grad_b = np.sum(grad_logits, axis=0, keepdims=True)  # shape: (1, n_classes)
        
        # Gradient descent update
        W -= lr * grad_W
        b -= lr * grad_b

    return W, b, loss_history


###############################################################################
# 6. MULTI-CLASS LOGISTIC REGRESSION: STOCHASTIC GRADIENT DESCENT
###############################################################################
def train_logreg_sgd(X_train, y_train_onehot, lr=0.1, epochs=100, batch_size=1):
    """
    X_train: (n_samples, n_features)
    y_train_onehot: (n_samples, n_classes)
    lr: learning rate
    epochs: number of passes through the dataset
    batch_size: how many samples per gradient update (default=1)
    
    returns: (W, b, loss_history)
    """
    n_samples, n_features = X_train.shape
    n_classes = y_train_onehot.shape[1]
    
    # Initialize weights and bias
    W = np.random.randn(n_features, n_classes) * 0.01
    b = np.zeros((1, n_classes))
    
    loss_history = []
    
    for epoch in range(epochs):
        # Shuffle the data for SGD
        indices = np.random.permutation(n_samples)
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train_onehot[indices]
        
        # Iterate over mini-batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = start_idx + batch_size
            xb = X_train_shuffled[start_idx:end_idx]
            yb = y_train_shuffled[start_idx:end_idx]
            
            # Forward pass
            logits = xb.dot(W) + b  # shape: (batch_size, n_classes)
            probs = softmax(logits)
            
            # Backprop
            grad_logits = (probs - yb) / batch_size
            grad_W = xb.T.dot(grad_logits)
            grad_b = np.sum(grad_logits, axis=0, keepdims=True)
            
            # Update
            W -= lr * grad_W
            b -= lr * grad_b
        
        # Compute loss on the entire training set for tracking
        logits_full = X_train.dot(W) + b
        probs_full = softmax(logits_full)
        loss = cross_entropy_loss(y_train_onehot, probs_full)
        loss_history.append(loss)
    
    return W, b, loss_history


###############################################################################
# 7. EVALUATION: CONFUSION MATRIX AND CLASSIFICATION METRICS
###############################################################################
def build_confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    return cm

def classification_metrics_from_cm(cm):
    """
    Given a confusion matrix, compute:
    - Accuracy
    - Precision (macro)
    - Recall (macro)
    - F1-score (macro)
    """
    num_classes = cm.shape[0]
    
    # Accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    
    # Precision and Recall for each class
    precisions = []
    recalls = []
    
    for c in range(num_classes):
        # Precision: TP / (TP + FP)
        #   TP = cm[c, c]
        #   FP = sum(cm[:, c]) - cm[c, c]
        tp = cm[c, c]
        fp = np.sum(cm[:, c]) - tp
        if tp + fp == 0:
            precision_c = 0.0
        else:
            precision_c = tp / (tp + fp)
        precisions.append(precision_c)
        
        # Recall: TP / (TP + FN)
        #   FN = sum(cm[c, :]) - cm[c, c]
        fn = np.sum(cm[c, :]) - tp
        if tp + fn == 0:
            recall_c = 0.0
        else:
            recall_c = tp / (tp + fn)
        recalls.append(recall_c)
    
    precision_macro = np.mean(precisions)
    recall_macro = np.mean(recalls)
    
    # F1-score (macro)
    if precision_macro + recall_macro == 0:
        f1_macro = 0.0
    else:
        f1_macro = 2 * (precision_macro * recall_macro) / (precision_macro + recall_macro)
    
    return accuracy, precision_macro, recall_macro, f1_macro


###############################################################################
# 8. MAIN SCRIPT: RUN EVERYTHING
###############################################################################

def main():
    # Load and split data
    X_train, y_train, X_test, y_test = load_and_split_data(test_ratio=0.4, random_seed=42)
    num_classes = len(np.unique(y_train))
    
    # One-hot encode the training labels for cross-entropy
    y_train_onehot = one_hot_encode(y_train, num_classes)
    
    # We'll run the pipeline for 3 different preprocessing methods:
    methods = ['unnormalized', 'standard', 'minmax']
    
    for method in methods:
        print("\n=============================")
        print(f"  Preprocessing Method: {method}")
        print("=============================")
        
        # Preprocess data
        X_train_prep, X_test_prep = get_preprocessed_data(X_train, X_test, method=method)
        
        # --------------------
        # (a) Gradient Descent
        # --------------------
        print("\n--- Gradient Descent ---")
        W_gd, b_gd, loss_history_gd = train_logreg_gd(X_train_prep, y_train_onehot, lr=0.1, epochs=100)
        
        # Evaluate on test set
        logits_test_gd = X_test_prep.dot(W_gd) + b_gd
        probs_test_gd = softmax(logits_test_gd)
        y_pred_gd = np.argmax(probs_test_gd, axis=1)
        
        # Build confusion matrix
        cm_gd = build_confusion_matrix(y_test, y_pred_gd, num_classes)
        accuracy, precision, recall, f1 = classification_metrics_from_cm(cm_gd)
        
        print("Confusion Matrix (GD):\n", cm_gd)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")
        
        # Plot loss curve for gradient descent
        plt.figure()
        plt.plot(loss_history_gd, label='Gradient Descent')
        plt.title(f"Loss Curve (GD) - {method}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.show()
        
        # --------------------
        # (b) Stochastic Gradient Descent
        # --------------------
        print("\n--- Stochastic Gradient Descent ---")
        W_sgd, b_sgd, loss_history_sgd = train_logreg_sgd(X_train_prep, y_train_onehot,
                                                          lr=0.1, epochs=100, batch_size=1)
        
        # Evaluate on test set
        logits_test_sgd = X_test_prep.dot(W_sgd) + b_sgd
        probs_test_sgd = softmax(logits_test_sgd)
        y_pred_sgd = np.argmax(probs_test_sgd, axis=1)
        
        # Build confusion matrix
        cm_sgd = build_confusion_matrix(y_test, y_pred_sgd, num_classes)
        accuracy_sgd, precision_sgd, recall_sgd, f1_sgd = classification_metrics_from_cm(cm_sgd)
        
        print("Confusion Matrix (SGD):\n", cm_sgd)
        print(f"Accuracy:  {accuracy_sgd:.4f}")
        print(f"Precision: {precision_sgd:.4f}")
        print(f"Recall:    {recall_sgd:.4f}")
        print(f"F1-score:  {f1_sgd:.4f}")
        
        # Plot loss curve for stochastic gradient descent
        plt.figure()
        plt.plot(loss_history_sgd, color='orange', label='Stochastic GD')
        plt.title(f"Loss Curve (SGD) - {method}")
        plt.xlabel("Epoch")
        plt.ylabel("Cross-Entropy Loss")
        plt.legend()
        plt.show()


if __name__ == "__main__":
    main()
