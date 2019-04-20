import pystan
import pickle
import argparse
import os
import pandas as pd
import numpy as np
import json


with open('SETTINGS.json', 'r') as f:
    SETTINGS = json.load(f)


def create_stan_model():
    '''
    Compile model for stan
    '''
    model_code = '''
    /*
    pairwise logistic regression model of winning the game

    */

    data {
      int<lower=0> N_teams;
      int<lower=0> N;  // number of games in regular season
      int<lower=0> N_tourney;  // number of games in tournament
      int<lower=1, upper=N_teams> j_team[N + N_tourney];  // index for team 1
      int<lower=1, upper=N_teams> k_team[N + N_tourney];  // index for team 2

      // features
      real x1[N + N_tourney];  // score_mean_team1 - score_opp_team2
      real x2[N + N_tourney];  // score_opp_team1 - score_mean_team2

      // response variables
      int<lower=0, upper=1> team1win[N];
      real y[N];  // scorediff
    }

    transformed data {

    }

    parameters {
      real alpha[N_teams];  // team effect
      real beta[N_teams];  // team scorediff effect
      real<lower=0> sigma_y;  // std for point difference
      real<lower=0> sigma_alpha;  // std for team levels
      real<lower=0> sigma_beta;  // std for scorediff effect
      real<lower=0> nu; // degrees of freedom for T distr
      real<lower=0, upper=20> mult; // effect multiplier between two models
    }

    transformed parameters {
      real<lower=0, upper=1> pi[N_tourney];
      vector[N_tourney] eta_tourney;  // linear predictor

      for(n in 1:N_tourney)
        eta_tourney[n] = alpha[j_team[N+n]] - alpha[k_team[N+n]]
                         + beta[j_team[N+n]] * x1[N+n]
                         + beta[k_team[N+n]] * x2[N+n];

      // probability that team1 wins
      for(n in 1:N_tourney) {
        pi[n] = inv_logit(eta_tourney[n]);
      }
    }

    model {
      vector[N] eta;  // linear predictor

      for(n in 1:N)
        eta[n] = alpha[j_team[n]] - alpha[k_team[n]]
                 + beta[j_team[n]] * x1[n]
                 + beta[k_team[n]] * x2[n];

      sigma_alpha ~ normal(0, 20);
      sigma_beta ~ normal(0, 20);
      alpha ~ normal(0, sigma_alpha); // team levels
      beta ~ normal(0, sigma_beta); // team levels

      nu ~ gamma(2, 0.1);

      // likelihoods
      team1win ~ bernoulli_logit(eta);
      y ~ student_t(nu, mult * eta, sigma_y);

    }

    generated quantities {

    }
    '''
    sm = pystan.StanModel(model_code=model_code)
    return sm


def create_stan_data(train_data_file, test_data_file):
    train = pd.read_csv(train_data_file)
    test = pd.read_csv(test_data_file)
    data = pd.concat([train, test]).reset_index()
    # teams
    teams = set(data['team1'].unique()).union(data['team2'].unique())
    team_f2id = dict(enumerate(teams, 1))  # start from 1 for stan's one-based indexing
    team_id2f = {v:k for k, v in team_f2id.items()}
    # data dict for stan
    stan_data = {
        'N_teams': len(teams),
        'N': (data.tourney == 0).sum(),
        'N_tourney': (data.tourney == 1).sum(),
        'j_team': data['team1'].map(team_id2f).values,
        'k_team': data['team2'].map(team_id2f).values,
        'x1': data['score_team_mean1'] - data['score_opp_mean2'],
        'x2': data['score_opp_mean1'] - data['score_team_mean2'],
        'team1win': data.loc[data.tourney == 0, 'team1win'].values,
        'y': data.loc[data.tourney == 0, 'score1'] - data.loc[data.tourney == 0, 'score2'],
    }
    return stan_data, data


if __name__ == '__main__':
    ##################################################
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='logistic')
    parser.add_argument('--num_iter', type=int, default=500)
    parser.add_argument('--num_chains', type=int, default=1)
    args = parser.parse_args()
    ##################################################
    # create model and fit directories
    model_directory = os.path.join(SETTINGS['MODEL_CHECKPOINT_DIR'],
                                   args.model_name)
    os.makedirs(model_directory, exist_ok=True)
    ##################################################
    # file names for serialized model and prediction
    model_fname = os.path.join(model_directory, 'model.pkl')
    fit_fname = os.path.join(model_directory, 'fit.pkl')
    prediction_fname = os.path.join(model_directory, 'prediction.csv')
    ##################################################
    # compile model
    stan_data, data = create_stan_data(SETTINGS['TRAIN_DATA_CLEAN_PATH'],
                                       SETTINGS['TEST_DATA_CLEAN_PATH'])
    if os.path.exists(model_fname):
        # load pickled model
        with open(model_fname, 'rb') as f:
            pickle_data = pickle.load(f)
            sm = pickle_data['sm']
    else:
        # compile model - takes a minute
        sm = create_stan_model()
        with open(model_fname, "wb") as f:
            pickle.dump({'sm': sm}, f, protocol=-1)
    ##################################################
    # sample from posterior (fit model)
    fit = sm.sampling(data=stan_data, iter=args.num_iter, chains=args.num_chains)
    with open(fit_fname, "wb") as f:
        pickle.dump({'sm': sm, 'fit': fit}, f, protocol=-1)
    ##################################################
    # extract model parameters and predictions
    la = fit.extract()
    alpha = la['alpha']
    pi = la['pi']
    # prediction
    y_pred = np.median(pi, axis=0)
    ID = data.loc[data['tourney'] == 1, 'ID'].values
    df_pred = pd.DataFrame({'ID':ID, 'Pred':y_pred})
    ##################################################
    # save prediction csv
    df_pred[['ID', 'Pred']].to_csv(prediction_fname, index=False)
