import pandas as pd
import numpy as np
import os
import json


with open('SETTINGS.json', 'r') as f:
    SETTINGS = json.load(f)

RAW_DATA_DIR = SETTINGS['RAW_DATA_DIR']


def get_train_data(season=None, detailed=False):
    detail = 'Detailed' if detailed else 'Compact'
    ##################################################
    # read data
    ##################################################
    RegularSeasonResults = pd.read_csv(
        os.path.join(RAW_DATA_DIR, 'RegularSeason{}Results.csv'.format(detail)))
    NCAATourneyResults = pd.read_csv(
        os.path.join(RAW_DATA_DIR, 'NCAATourney{}Results.csv'.format(detail)))
    NCAATourneySeeds = pd.read_csv(
        os.path.join(RAW_DATA_DIR, 'NCAATourneySeeds.csv'))
    TeamConferences = pd.read_csv(
        os.path.join(RAW_DATA_DIR, 'TeamConferences.csv'))
    Conferences = pd.read_csv(
        os.path.join(RAW_DATA_DIR, 'Conferences.csv'))
    Teams = pd.read_csv(
        os.path.join(RAW_DATA_DIR, 'Teams.csv'))
    TeamConferences = (pd.merge(TeamConferences, Conferences, on='ConfAbbrev')
                       .rename({'Description': 'conf_descr'}, axis=1))
    SampleSubmissionStage2 = pd.read_csv(
        os.path.join(RAW_DATA_DIR, '../SampleSubmissionStage2.csv'))
    tourney2019 = SampleSubmissionStage2['ID'].str.split('_', expand=True).astype(int)
    # tourney2019.index = SampleSubmissionStage2['ID']
    tourney2019.columns = ['Season', 'WTeamID', 'LTeamID']
    NCAATourneyResults = pd.concat([NCAATourneyResults, tourney2019], sort=True)
    # tourney2019 = tourney2019.reset_index()
    ##################################################
    # process data
    ##################################################
    NCAATourneySeeds['seednum'] = NCAATourneySeeds['Seed'].str.slice(1, 3).astype(int)
    RegularSeasonResults['tourney'] = 0
    NCAATourneyResults['tourney'] = 1
    # combine regular and tourney data
    data = pd.concat([RegularSeasonResults, NCAATourneyResults], sort=True)
    if season:
        data = data[data.Season == season]  # filter season
    ##################################################
    # team1: team with lower id
    data['team1'] = (data['WTeamID'].where(data['WTeamID'] < data['LTeamID'],
                                           data['LTeamID']))
    # team2: team with higher id
    data['team2'] = (data['WTeamID'].where(data['WTeamID'] > data['LTeamID'],
                                           data['LTeamID']))
    data['score1'] = data['WScore'].where(data['WTeamID'] < data['LTeamID'], data['LScore'])
    data['score2'] = data['WScore'].where(data['WTeamID'] > data['LTeamID'], data['LScore'])
    boxscore_stats = ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
                      'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF',]
    if detailed:
        for stat in boxscore_stats:
            data[stat + '_team1'] = data['W' + stat].where(data['WTeamID'] < data['LTeamID'],
                                                      data['L' + stat])
            data[stat + '_team2'] = data['W' + stat].where(data['WTeamID'] > data['LTeamID'],
                                                      data['L' + stat])
        data = data.drop(['W'+stat for stat in boxscore_stats], axis=1)
        data = data.drop(['L'+stat for stat in boxscore_stats], axis=1)
    data['loc'] = (data['WLoc']
                   .where(data['WLoc'] != 'H', data['WTeamID'])
                   .where(data['WLoc'] != 'A', data['LTeamID'])
                   .where(data['WLoc'] != 'N', 0))  # 0 if no home court
    data['team1win'] = np.where(data['WTeamID'] == data['team1'], 1, 0)
    ##################################################
    # get tourney seeds
    data = (data
            .pipe(pd.merge, NCAATourneySeeds,
                  left_on=['Season', 'team1'], right_on=['Season', 'TeamID'],
                  how='left')
            .pipe(pd.merge, NCAATourneySeeds,
                  left_on=['Season', 'team2'], right_on=['Season', 'TeamID'],
                  how='left', suffixes=('1', '2'))
            )
    ##################################################
    # get conferences
    data = (data
            .pipe(pd.merge, TeamConferences,
                  left_on=['Season', 'team1'], right_on=['Season', 'TeamID'],
                  how='left')
            .pipe(pd.merge, TeamConferences,
                  left_on=['Season', 'team2'], right_on=['Season', 'TeamID'],
                  how='left', suffixes=('1', '2'))
            )
    ##################################################
    # get team names
    data = (data
            .pipe(pd.merge, Teams,
                  left_on=['team1'], right_on=['TeamID'],
                  how='left')
            .pipe(pd.merge, Teams,
                  left_on=['team2'], right_on=['TeamID'],
                  how='left', suffixes=('1', '2'))
            )
    # calculate seed diff
    data['seeddiff'] = data['seednum2'] - data['seednum1']
    data = data.drop(['TeamID1', 'TeamID2', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc'], axis=1)
    data.columns = data.columns.str.lower()
    data['ID'] = (data[['season', 'team1', 'team2']].astype(str)
                  .apply(lambda x: '_'.join(x), axis=1))
    return data


def get_boxscore_dataset_v1(season=None, detailed=False, final_prediction=False):
    '''
    Extend train_data_v1 with seasonwise mean/std boxscore columns for each team and opponent
    '''
    if final_prediction:
        data_sub = pd.read_csv(
            os.path.join(RAW_DATA_DIR, '../SampleSubmissionStage2.csv'))
        data = data_sub['ID'].str.split('_', expand=True).astype(int)
        data.index = data_sub['ID']
        data.columns = ['season', 'team1', 'team2']
        data = data.reset_index()
    else:
        data = get_train_data_v1(season=season, detailed=detailed) # main data
    ##################################################
    # regular season boxscore data
    ##################################################
    RegularSeasonDetailedResults = pd.read_csv(
        os.path.join(RAW_DATA_DIR, 'RegularSeasonDetailedResults.csv'))
    ##################################################
    # column processing
    ##################################################
    cols = RegularSeasonDetailedResults.columns
    w_cols = (cols.str.slice(0, 1) == 'W') & (~cols.isin(['WLoc']))
    l_cols = cols.str.slice(0, 1) == 'L'
    box_colnames = cols[w_cols].str.slice(1)  # remove 'W' and 'L'
    # for reversing W columns with L cols
    reverse_dict = dict(zip(list('W' + box_colnames) + list('L' + box_colnames),
                            list('L' + box_colnames) + list('W' + box_colnames)))
    # for converting W and L boxstats to team and opponent boxstats
    rename_dict = dict(zip(list('W' + box_colnames) + list('L' + box_colnames),
                           list(box_colnames + '_team') + list(box_colnames + '_opp')))
    ##################################################
    # stack the original and reversed dataframes
    ##################################################
    RegularSeasonDetailedResultsStacked = pd.concat(
        [RegularSeasonDetailedResults,
         RegularSeasonDetailedResults.rename(reverse_dict, axis=1)],
        sort=True).rename(rename_dict, axis=1)
    n = RegularSeasonDetailedResults.shape[0]
    RegularSeasonDetailedResultsStacked['win'] = np.array([True] * n + [False] * n)
    ##################################################
    # calculate boxscore stats
    ##################################################
    df_boxstat = (RegularSeasonDetailedResultsStacked[list(rename_dict.values()) + ['Season']]
                  .groupby(['Season', 'TeamID_team'])
                  .agg(['mean', 'std']))
    df_boxstat.columns = ['_'.join(col).strip() for col in df_boxstat.columns.values]
    df_boxstat.columns = df_boxstat.columns.str.lower()
    drop_cols = df_boxstat.columns[df_boxstat.columns.str.contains('teamid_opp')]
    df_boxstat = df_boxstat.drop(drop_cols, axis=1)
    df_boxstat.index.names = ['Season', 'TeamID']
    ##################################################
    # merge with main data
    ##################################################
    data = (data
            .pipe(pd.merge, df_boxstat,
                  left_on=['season', 'team1'], right_index=True,
                  how='left')
            .pipe(pd.merge, df_boxstat,
                  left_on=['season', 'team2'], right_index=True,
                  how='left', suffixes=('1', '2'))
            )
    return data


if __name__ == '__main__':
    TRAIN_DATA_CLEAN_PATH = SETTINGS['TRAIN_DATA_CLEAN_PATH']
    TEST_DATA_CLEAN_PATH = SETTINGS['TEST_DATA_CLEAN_PATH']
    data = get_boxscore_dataset_v1(season=2019)
    data.loc[data['tourney'] == 0].to_csv(TRAIN_DATA_CLEAN_PATH,
                                          index=False)
    data.loc[data['tourney'] == 1].to_csv(TEST_DATA_CLEAN_PATH,
                                          index=False)
