import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
from src import utils
from src.data import make_dataset
from matplotlib import pyplot as plt
import seaborn as sns
import os
import json


with open('SETTINGS.json', 'r') as f:
    SETTINGS = json.load(f)


if __name__ == '__main__':
    ##################################################
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.77)
    parser.add_argument('--prob_adj', type=int, default=0.14)
    args = parser.parse_args()
    ##################################################
    # load logistic regression model
    prediction = pd.read_csv(os.path.join(SETTINGS['MODEL_CHECKPOINT_DIR'],
                                          'logistic/prediction.csv'))
    logistic_prediction = prediction['Pred']
    logistic_confidence = (prediction['Pred'] - 0.5).abs() + 0.5
    ##################################################
    # adjust low confidence predictions
    is_confident = logistic_confidence >= args.threshold
    prediction['submission1'] = logistic_prediction.where(
        is_confident,
        np.ones(prediction.shape[0]) * 0.5 + args.prob_adj)
    prediction['submission2'] = logistic_prediction.where(
        is_confident,
        np.ones(prediction.shape[0]) * 0.5 - args.prob_adj)
    ##################################################
    # save submissions
    final_model_names = ['submission1', 'submission2']
    rename_dict = dict(zip(final_model_names, ['Pred', 'Pred']))
    for i, m in enumerate(final_model_names, 1):
        fname = 'submission{}.csv'.format(i)
        pred = prediction[['ID', m]].rename(rename_dict, axis=1)
        pred.to_csv(os.path.join(SETTINGS['SUBMISSION_DIR'], fname),
                    index=False)
