# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from utils import read_json, convert_to_ordinal

test_dir = Path('/crownchallenge/test')
output_dir = Path('/crownchallenge/output/teamname_task1')


def get_balanced_accuracy(y_true, prediction):
    return balanced_accuracy_score(y_true, prediction)


def main():
    """Main function"""
    result_files = list(output_dir.glob('*/result_Lippert.json'))
    if len(result_files) != 300:
        raise RuntimeError('Wrong number of result files')

    predictions_ant = np.zeros(300, dtype=int)
    test_ant = np.zeros(300, dtype=int)
    
    predictions_pos = np.zeros(300, dtype=int)
    test_pos = np.zeros(300, dtype=int)
    
    for i, file in enumerate(result_files):
        
        # Read result file
        prediction = read_json(file)
        
        # Read test file
        test_file = test_dir / file.parent.stem / 'Lippert_classes.json'
        label = read_json(test_file)
    
        #        
        predictions_ant[i] = convert_to_ordinal(prediction['Anterior class'])
        test_ant[i] = convert_to_ordinal(label['Anterior class'])
        
        predictions_pos[i] = convert_to_ordinal(prediction['Posterior class'])
        test_pos[i] = convert_to_ordinal(label['Posterior class'])

    # Print results
    ba_anterior = get_balanced_accuracy(test_ant, predictions_ant)
    ba_posterior = get_balanced_accuracy(test_pos, predictions_pos)
    
    print(f'BA (ant): {ba_anterior}')    
    print(f'BA (pos): {ba_posterior}')


if __name__ == "__main__":
    main()    
