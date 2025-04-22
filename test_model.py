import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import pandas as pd
from typing import List, Dict, Union, Tuple, Optional


def evaluate_model(model: tf.keras.Model, 
                  dataset: tf.data.Dataset, 
                  steps: int, 
                  class_names: List[str],
                  history: Optional[tf.keras.callbacks.History] = None,
                  title: str = "Model Evaluation Results") -> Dict:
    """
    Comprehensively evaluate model performance and visualize results
    
    Parameters:
    - model: Trained Keras model
    - dataset: Evaluation dataset (usually test set)
    - steps: Evaluation steps (usually test set samples/batch size limit)
    - class_names: List of class names
    - history: Model training history object (for overfitting/underfitting analysis)
    - title: Results visualization title
    
    Returns:
    - Dictionary containing various evaluation metrics
    """
    # 1. Get prediction results
    print(f"processing...")
    y_pred_prob = model.predict(dataset, steps=steps)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Get true labels
    y_true = []
    for _, labels in dataset.take(steps):
        y_true.extend(labels.numpy())
    y_true = np.array(y_true[:len(y_pred)])  # Truncate to same length
    
    # 2. Calculate basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names)))
    
    # Calculate macro average and weighted average
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro')
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted')
    
    # 3. Create results dictionary
    results = {
        'accuracy': accuracy,
        'class_precision': dict(zip(class_names, precision)),
        'class_recall': dict(zip(class_names, recall)),
        'class_f1': dict(zip(class_names, f1)),
        'class_support': dict(zip(class_names, support)),
        'macro_precision': macro_precision,
        'macro_recall': macro_recall, 
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1
    }
    
    # 4. Visualize results
    visualize_results(y_true, y_pred, y_pred_prob, class_names, 
                      history=history, title=title)
    
    # 5. Print detailed report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    print(f"\nOverall Accuracy: {accuracy:.4f}")
    print(f"Macro-average F1 Score: {macro_f1:.4f}")
    print(f"Weighted-average F1 Score: {weighted_f1:.4f}")
    
    return results


def visualize_results(y_true: np.ndarray, 
                     y_pred: np.ndarray, 
                     y_pred_prob: np.ndarray,
                     class_names: List[str],
                     history: Optional[tf.keras.callbacks.History] = None,
                     title: str = "Model Evaluation Result"):
    """
    Visualize model evaluation results
    
    Parameters:
    - y_true: True label array
    - y_pred: Predicted label array
    - y_pred_prob: Prediction probability array
    - class_names: List of class names
    - history: Model training history object
    - title: Visualization title
    """
    # Set up fonts for English labels
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # Create subplot layout
    n_plots = 3 if history is not None else 2
    fig = plt.figure(figsize=(15, 5 * n_plots))
    fig.suptitle(title, fontsize=16)
    
    # 1. Plot confusion matrix
    ax1 = fig.add_subplot(n_plots, 2, 1)
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Normalized Confusion Matrix')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # 2. Plot accuracy, precision, recall and F1 score for each class
    ax2 = fig.add_subplot(n_plots, 2, 2)
    
    # Calculate metrics for each class
    metrics_per_class = {}
    for i, class_name in enumerate(class_names):
        true_positives = np.sum((y_true == i) & (y_pred == i))
        false_positives = np.sum((y_true != i) & (y_pred == i))
        false_negatives = np.sum((y_true == i) & (y_pred != i))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_per_class[class_name] = {
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }
    
    # Create DataFrame and visualize
    metrics_df = pd.DataFrame(metrics_per_class).T
    metrics_df.plot(kind='bar', ax=ax2)
    ax2.set_title('Performance Metrics by Class')
    ax2.set_ylabel('Score')
    ax2.set_ylim([0, 1])
    ax2.legend(loc='lower right')
    
    # 3. If training history is provided, plot learning curves for overfitting/underfitting analysis
    if history is not None:
        # Accuracy curve
        ax3 = fig.add_subplot(n_plots, 2, 3)
        ax3.plot(history.history['sparse_categorical_accuracy'], label='Train accuracy')
        ax3.plot(history.history['val_sparse_categorical_accuracy'], label='Var accuracy')
        ax3.set_title('Model Accuracy')
        ax3.set_ylabel('Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.legend(loc='lower right')
        
        # Loss curve
        ax4 = fig.add_subplot(n_plots, 2, 4)
        ax4.plot(history.history['loss'], label='Train loss')
        ax4.plot(history.history['val_loss'], label='Var loss')
        ax4.set_title('Model Loss')
        ax4.set_ylabel('Loss')
        ax4.set_xlabel('Epoch')
        ax4.legend(loc='upper right')
        
        # Overfitting/underfitting analysis
        ax5 = fig.add_subplot(n_plots, 2, 5)
        
        # Calculate gap between training and validation
        train_acc = history.history['sparse_categorical_accuracy']
        val_acc = history.history['val_sparse_categorical_accuracy']
        acc_diff = [train - val for train, val in zip(train_acc, val_acc)]
        
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        loss_diff = [val - train for train, val in zip(train_loss, val_loss)]
        
        epochs = range(1, len(train_acc) + 1)
        
        ax5.plot(epochs, acc_diff, label='Accuracy Gap (Train-Val)')
        ax5.plot(epochs, loss_diff, label='Loss Gap (Val-Train)')
        ax5.axhline(y=0.05, color='r', linestyle='--', label='Overfitting Threshold')
        ax5.set_title('Overfitting/Underfitting Analysis')
        ax5.set_ylabel('Gap')
        ax5.set_xlabel('Epoch')
        ax5.legend()
        
        # Add overfitting analysis text description
        ax6 = fig.add_subplot(n_plots, 2, 6)
        ax6.axis('off')
        
        max_acc_diff = max(acc_diff) if acc_diff else 0
        max_loss_diff = max(loss_diff) if loss_diff else 0
        
        if max_acc_diff > 0.1 or max_loss_diff > 0.1:
            status = "Overfitting"
            explanation = "Model performs significantly better on training data than validation data, indicating overfitting."
            recommendation = "Recommendation: Increase regularization, use more data, reduce model complexity, or use early stopping."
        elif max(train_acc) < 0.7 and max(val_acc) < 0.7:
            status = "Underfitting"
            explanation = "Model performs poorly on both training and validation data, suggesting insufficient model capacity."
            recommendation = "Recommendation: Increase model complexity, reduce regularization, train longer, or adjust learning rate."
        else:
            status = "Good Fit"
            explanation = "Model performs similarly on training and validation sets with no significant overfitting or underfitting."
            recommendation = "Model is well-trained and ready for testing and deployment."
        
        text = f"Model Status: {status}\n\n{explanation}\n\n{recommendation}"
        ax6.text(0.1, 0.5, text, fontsize=12, verticalalignment='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def detect_overfitting(history: tf.keras.callbacks.History) -> str:
    """
    Analyze training history, detect overfitting/underfitting
    
    Parameters:
    - history: Model training history object
    
    Returns:
    - Overfitting/underfitting status analysis string
    """
    train_acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Calculate gaps in late epochs
    late_epochs = len(train_acc) // 3
    late_acc_diff = np.mean([t - v for t, v in zip(train_acc[-late_epochs:], val_acc[-late_epochs:])])
    late_loss_diff = np.mean([v - t for t, v in zip(train_loss[-late_epochs:], val_loss[-late_epochs:])])
    
    # Determine overfitting/underfitting
    if late_acc_diff > 0.1 or late_loss_diff > 0.1:
        status = "Overfitting"
        level = "Severe" if (late_acc_diff > 0.2 or late_loss_diff > 0.2) else "Mild"
        detail = f"Training accuracy exceeds validation by {late_acc_diff:.4f}, validation loss exceeds training by {late_loss_diff:.4f}"
        suggestion = ("Recommendation: Increase dropout ratio, strengthen L2 regularization, reduce network layers or neurons, "
                     "use early stopping, increase data augmentation, or collect more training data")
    elif max(train_acc) < 0.7 and max(val_acc) < 0.7:
        status = "Underfitting"
        level = "Severe" if max(train_acc) < 0.6 else "Mild"
        detail = f"Both training and validation accuracy are low, with maximum training accuracy of {max(train_acc):.4f}"
        suggestion = ("Recommendation: Increase network complexity, extend training duration, reduce regularization, "
                     "adjust learning rate, consider using pre-trained models or feature engineering")
    else:
        status = "Good Fit"
        detail = f"Training and validation metrics are close, with final validation accuracy of {val_acc[-1]:.4f}"
        suggestion = "Model is well-trained and ready for deployment or further testing"
        level = ""
    
    # Trend analysis
    val_acc_trend = val_acc[-1] - np.mean(val_acc[-(late_epochs+1):-1])
    trend_text = "Validation accuracy still rising" if val_acc_trend > 0.01 else (
                 "Validation accuracy stabilized" if abs(val_acc_trend) <= 0.01 else "Validation accuracy declining")
    
    return f"Model Status: {level+' ' if level else ''}{status}\n" \
           f"Details: {detail}\n" \
           f"Trend: {trend_text}\n" \
           f"{suggestion}"


def compare_models(models: List[tf.keras.Model], 
                  names: List[str], 
                  test_dataset: tf.data.Dataset, 
                  steps: int,
                  class_names: List[str]) -> None:
    """
    Compare performance of multiple models
    
    Parameters:
    - models: List of trained models
    - names: List of model names
    - test_dataset: Test dataset
    - steps: Test steps
    - class_names: List of class names
    """
    # Get true labels
    y_true = []
    for _, labels in test_dataset.take(steps):
        y_true.extend(labels.numpy())
    y_true = np.array(y_true)
    
    # Collect results
    results = []
    
    for model, name in zip(models, names):
        print(f"\nEvaluating model: {name}")
        y_pred_prob = model.predict(test_dataset, steps=steps)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Truncate to same length
        y_pred = y_pred[:len(y_true)]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro')
        
        results.append({
            'name': name,
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'y_pred': y_pred
        })
        
        print(f"Accuracy: {accuracy:.4f}, F1 Score: {macro_f1:.4f}")
    
    # Visualize comparison
    plt.figure(figsize=(12, 6))
    metrics = ['accuracy', 'macro_precision', 'macro_recall', 'macro_f1']
    metrics_names = ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 Score (Macro)']
    
    bar_width = 0.2
    index = np.arange(len(metrics))
    
    for i, result in enumerate(results):
        values = [result[metric] for metric in metrics]
        plt.bar(index + i * bar_width, values, bar_width, label=result['name'])
    
    plt.xlabel('Evaluation Metrics')
    plt.ylabel('Score')
    plt.title('Model Performance Comparison')
    plt.xticks(index + bar_width * (len(results) - 1) / 2, metrics_names)
    plt.ylim([0, 1])
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Compare confusion matrices
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 5))
    if len(models) == 1:
        axes = [axes]
        
    for i, (result, ax) in enumerate(zip(results, axes)):
        cm = confusion_matrix(y_true, result['y_pred'])
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_title(f'{result["name"]} - Confusion Matrix')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()


def model_diagnostics(model: tf.keras.Model, 
                     test_dataset: tf.data.Dataset,
                     steps: int,
                     class_names: List[str]) -> None:
    """
    Perform detailed diagnostics on model performance, analyze misclassified samples
    
    Parameters:
    - model: Trained model
    - test_dataset: Test dataset
    - steps: Test steps
    - class_names: List of class names
    """
    # Collect predictions and true labels
    all_samples = []
    all_labels = []
    for samples, labels in test_dataset.take(steps):
        all_samples.append(samples.numpy())
        all_labels.append(labels.numpy())
    
    all_samples = np.vstack(all_samples)
    all_labels = np.concatenate(all_labels)
    
    # Get prediction results
    y_pred_prob = model.predict(test_dataset, steps=steps)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Truncate to same length
    all_samples = all_samples[:len(y_pred)]
    all_labels = all_labels[:len(y_pred)]
    
    # Calculate prediction certainty (maximum probability)
    certainty = np.max(y_pred_prob, axis=1)
    
    # Find misclassified samples
    misclassified = all_labels != y_pred
    
    # Sort misclassified samples by certainty
    misclassified_indices = np.where(misclassified)[0]
    misclassified_certainty = certainty[misclassified]
    
    # Find most certain errors and least certain errors
    if len(misclassified_indices) > 0:
        most_certain_errors = misclassified_indices[np.argsort(-misclassified_certainty)][:5]
        least_certain_errors = misclassified_indices[np.argsort(misclassified_certainty)][:5]
        
        print("\nMisclassification Analysis:")
        print(f"Total misclassified samples: {np.sum(misclassified)}/{len(all_labels)} ({np.mean(misclassified)*100:.2f}%)")
        
        # Analyze errors by class
        print("\nError rate by class:")
        for i, class_name in enumerate(class_names):
            class_samples = all_labels == i
            class_errors = np.logical_and(class_samples, misclassified)
            error_rate = np.sum(class_errors) / np.sum(class_samples) if np.sum(class_samples) > 0 else 0
            print(f"  {class_name}: {error_rate*100:.2f}% ({np.sum(class_errors)}/{np.sum(class_samples)})")
        
        # Analyze most common confusion pairs
        print("\nMost common confusion pairs:")
        cm = confusion_matrix(all_labels, y_pred)
        np.fill_diagonal(cm, 0)  # Ignore diagonal (correct classifications)
        top_confusions = []
        
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                if i != j and cm[i, j] > 0:
                    top_confusions.append((i, j, cm[i, j]))
        
        top_confusions.sort(key=lambda x: x[2], reverse=True)
        for true_class, pred_class, count in top_confusions[:5]:
            print(f"  {class_names[true_class]} misclassified as {class_names[pred_class]}: {count} times")
        
        # Analyze by prediction certainty
        certainty_bins = [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0]
        print("\nError distribution by prediction certainty:")
        for i in range(len(certainty_bins) - 1):
            bin_mask = (certainty >= certainty_bins[i]) & (certainty < certainty_bins[i+1])
            bin_total = np.sum(bin_mask)
            bin_errors = np.sum(misclassified & bin_mask)
            bin_error_rate = bin_errors / bin_total if bin_total > 0 else 0
            print(f"  Certainty {certainty_bins[i]:.2f}-{certainty_bins[i+1]:.2f}: "
                  f"{bin_error_rate*100:.2f}% ({bin_errors}/{bin_total})")
    else:
        print("No misclassified samples found!")


