# -*- coding: utf-8 -*-

import json

import numpy as np


def read_json(path):
    """Returns data from JSON file"""
    with open(path) as f:
        data = json.load(f)
    
    return data


def convert_to_ordinal(char):
    """Returns ordinal of ASCII character"""
    return ord(char) - 96


def exclude_missing_values(prediction_array, y_true_array):
    """Removes missing values from evaluation.
    
    This function is used to exclude predictions with missing values in the test labels.
    When evaluating the performance of teams' predictions, any case where the true 
    label is missing will not be taken into account and will not contribute to their 
    overall score. The predicted values for these cases are disregarded."""
    
    missing_values = [[ind for ind, y in enumerate(y_true) if y == 0 or np.isnan(y)] for y_true in y_true_array]
    
    prediction_array_cleaned = []
    y_true_array_cleaned = []
    
    for i, curr_pred in enumerate(prediction_array):
        curr_missing = missing_values[i]
        
        prediction_array_cleaned.append(np.delete(curr_pred, curr_missing))
        y_true_array_cleaned.append(np.delete(y_true_array[i], curr_missing))
        
        assert len(prediction_array_cleaned) == len(y_true_array_cleaned)
        
    return prediction_array_cleaned, y_true_array_cleaned
