#!/usr/bin/env python3
"""
Evaluation script for BERT classifier predictions.

This module evaluates the prediction results from the BERT classifier
by comparing predicted labels against ground truth labels.

Example:
    $ python predict_eval.py
"""

from typing import List, Tuple


# Configuration paths
INFERENCE_RESULT_PATH = 'output/test_results.tsv'
TEST_FILE_PATH = './dataset/test.csv'


def load_predictions(file_path: str) -> List[int]:
    """
    Load predictions from a TSV file.

    Args:
        file_path: Path to the prediction results file.

    Returns:
        A list of predicted labels (0 or 1).
    """
    predictions = []
    with open(file_path, 'r', encoding='utf-8') as file_handle:
        for line in file_handle:
            scores = [float(x) for x in line.strip().split()]
            # Get index of max score as the predicted label
            predicted_label = scores.index(max(scores))
            predictions.append(predicted_label)
    return predictions


def load_ground_truth(file_path: str) -> List[int]:
    """
    Load ground truth labels from a test file.

    Args:
        file_path: Path to the test file.

    Returns:
        A list of ground truth labels.
    """
    labels = []
    with open(file_path, 'r', encoding='utf-8') as file_handle:
        for line in file_handle:
            label = line.split('<>')[0].strip()
            labels.append(int(label))
    return labels


def calculate_accuracy(predictions: List[int], ground_truth: List[int]) -> float:
    """
    Calculate the accuracy of predictions.

    Args:
        predictions: List of predicted labels.
        ground_truth: List of ground truth labels.

    Returns:
        The accuracy as a float between 0 and 1.

    Raises:
        ValueError: If predictions and ground truth have different lengths.
    """
    if len(predictions) != len(ground_truth):
        raise ValueError(
            f"Length mismatch: predictions ({len(predictions)}) vs "
            f"ground_truth ({len(ground_truth)})"
        )
    
    correct_count = sum(
        1 for pred, true in zip(predictions, ground_truth) if pred == true
    )
    return correct_count / len(ground_truth)


def evaluate(inference_result_path: str, test_file_path: str) -> float:
    """
    Evaluate the inference results against ground truth labels.

    Args:
        inference_result_path: Path to the inference results TSV file.
        test_file_path: Path to the test file with ground truth labels.

    Returns:
        The accuracy of the predictions.
    """
    predictions = load_predictions(inference_result_path)
    ground_truth = load_ground_truth(test_file_path)
    
    print(f"Predictions: {predictions}")
    print(f"Ground Truth: {ground_truth}")
    
    accuracy = calculate_accuracy(predictions, ground_truth)
    correct_count = int(accuracy * len(ground_truth))
    print(f"Correct predictions: {correct_count}/{len(ground_truth)}")
    
    return accuracy


def main() -> None:
    """Main entry point for evaluation."""
    try:
        accuracy = evaluate(INFERENCE_RESULT_PATH, TEST_FILE_PATH)
        print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    except FileNotFoundError as e:
        print(f"Error: File not found - {e.filename}")
        print("Please ensure the inference results and test file exist.")
    except Exception as e:
        print(f"Error during evaluation: {e}")


if __name__ == "__main__":
    main()
