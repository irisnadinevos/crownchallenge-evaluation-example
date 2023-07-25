# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 11:15:34 2023

@author: iris
"""

import numpy as np

from pathlib import Path
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

from utils import read_json, exclude_missing_values

test_dir = ''

output_dir = Path('/output/crownchallenge/teamname_task2')


def main():
    """Main function"""
    
    # Diameters
    predictions, labels = get_results('results_diameters.json', 'artery_diameters.json', 15)
    
    mae = [get_MAE(predictions[i], labels[i]) for i in range(15)]
    pearson = [get_Pearson(predictions[i], labels[i]) for i in range(15)]
    
    print(f'MAE (diameters): {mae}')
    print(f'Pearson (diameters): {pearson}')
    
    # Angles
    prediction, label = get_results('results_angles.json', 'bifurcation_angles.json', 10)
    
    mae = [get_MAE(predictions[i], labels[i]) for i in range(10)]
    pearson = [get_Pearson(predictions[i], labels[i]) for i in range(10)]
    
    print(f'MAE (diameters): {mae}')
    print(f'Pearson (diameters): {pearson}')


def get_results(file_str, label_str, num_entries):
    """Returns the predictions and labels as NumPy array"""
    
    results_data = list(output_dir.glob(f'*/{file_str}'))
    assert len(results_data) == 300
    
    data_predicted = np.zeros((num_entries, 300))
    data_labels = np.zeros((num_entries, 300))
    
    for i, file in enumerate(results_data):
        # Read result file
        prediction = read_json(file)

        # Read test file
        test_file = test_dir / file.parent.stem / label_str
        label = read_json(test_file)

        # Get results
        for ii, d in enumerate(label.keys()):
            data_predicted[ii, i, ...] = prediction[d]
            data_labels[ii, i, ...] = label[d]
    
    
    # Exclude missing values in the test set from evaluation
    prediction_array, y_true_array = exclude_missing_values(data_predicted, data_labels)

    return prediction_array, y_true_array



def get_MAE(prediction, y_true):
    """Returns the mean absolute error (lower is better)"""
    
    return mean_absolute_error(prediction, y_true)


def get_Pearson(prediction, y_true):
    """Returns object with the Pearson correlation coefficient 
    (closer to 1 is better) and p-value"""
    
    return pearsonr(prediction, y_true)


if __name__ == "__main__":
    main()    
