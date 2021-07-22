# Tools for the analisys of Datti Machine data
import pandas as pd
from sklearn import preprocessing
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.random import seed
from sklearn.decomposition import PCA
from multiprocessing import Pool
import concurrent
from datetime import datetime as datetime_imported
sns.set(color_codes=True)
date_format = '%d/%m/%Y  %H:%M:%S'
date_format_2 = '%d/%m/%Y  %H:%M'
#date_format = '%Y-%m-%d %H:%M:%S'

def cov_matrix_calc(data, verbose=False):
    covariance_matrix = np.cov(data, rowvar=False)
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        if is_pos_def(inv_covariance_matrix):
            return covariance_matrix, inv_covariance_matrix
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")

def MahalanobisDist(inv_cov_matrix, mean_distr, data, verbose=False):
    inv_covariance_matrix = inv_cov_matrix
    vars_mean = mean_distr
    diff = data - vars_mean
    md = []
    for i in range(len(diff)):
        md.append(np.sqrt(diff[i].dot(inv_covariance_matrix).dot(diff[i])))
    return md

def MD_detectOutliers(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    outliers = []
    for i in range(len(dist)):
        if dist[i] >= threshold:
            outliers.append(i)  # index of the outlier
    return np.array(outliers)

def MD_threshold(dist, extreme=False, verbose=False):
    k = 3. if extreme else 2.
    threshold = np.mean(dist) * k
    return threshold

def is_pos_def(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def dist_and_thresh(X_train_PCA, X_test_PCA):
    data_train = np.array(X_train_PCA.values)
    data_test = np.array(X_test_PCA.values)
    cov_matrix, inv_cov_matrix = cov_matrix_calc(data_train)
    mean_distr = data_train.mean(axis=0)

    dist_test = MahalanobisDist(inv_cov_matrix, mean_distr, data_test, verbose=False)
    dist_train = MahalanobisDist(inv_cov_matrix, mean_distr, data_train, verbose=False)
    threshold = MD_threshold(dist_train, extreme = True)
    return dist_test, dist_train, threshold



def scale(dataset_train, dataset_test,components):
    scaler = preprocessing.MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(dataset_train),
                           columns=dataset_train.columns,
                           index=dataset_train.index)
    X_train.sample(frac=1)
    X_test = pd.DataFrame(scaler.transform(dataset_test),
                          columns=dataset_test.columns,
                          index=dataset_test.index)
    X_train_PCA, X_test_PCA = PCA_transformations(X_train,X_test,components)
    return X_train_PCA, X_test_PCA

def PCA_transformations(X_train, X_test, components):
    pca = PCA(n_components=components, svd_solver='full')

    X_train_PCA = pca.fit_transform(X_train)
    X_train_PCA = pd.DataFrame(X_train_PCA)
    X_train_PCA.index = X_train.index

    X_test_PCA = pca.transform(X_test)
    X_test_PCA = pd.DataFrame(X_test_PCA)
    X_test_PCA.index = X_test.index
    return X_train_PCA, X_test_PCA

    
def dists_and_anomalies(data_train, data_test, components=2):
    X_train_PCA, X_test_PCA = scale(data_train, data_test, components)
    dist_test, dist_train, threshold = dist_and_thresh(X_train_PCA, X_test_PCA)
    anomaly_train = anomaly_training(dist_train, threshold, X_train_PCA) 
    anomaly = anomaly_test(dist_test, threshold, X_test_PCA)
    anomaly_alldata = pd.concat([anomaly_train, anomaly])
    return dist_test, dist_train, threshold, anomaly_alldata

def anomaly_test(dist_test, threshold, X_test_PCA):
    anomaly = pd.DataFrame()
    anomaly['Mob dist']= dist_test
    anomaly['Thresh'] = threshold
    anomaly['Anomaly'] = anomaly['Mob dist'] > anomaly['Thresh']
    anomaly.index = X_test_PCA.index
    return anomaly

def anomaly_training(dist_train, threshold, X_train_PCA):
    anomaly_train = pd.DataFrame()
    anomaly_train['Mob dist']= dist_train
    anomaly_train['Thresh'] = threshold
    anomaly_train['Anomaly'] = anomaly_train['Mob dist'] > anomaly_train['Thresh']
    anomaly_train.index = X_train_PCA.index
    return anomaly_train

def plot_anomaly_metric(anomaly_alldata, scale_1=10, scale_2=6):
    tmp = anomaly_alldata.drop('Thresh', axis=1)
    tmp.plot(logy=True, figsize = (scale_1,scale_2), ylim = [1e-1,1e3], color = ['green','red'])

def plot_M_distance_squared(dist_train, scale=10):
    plt.figure()
    sns.distplot(np.square(dist_train),
                            bins=10,
                            kde=False)
    plt.xlim([0.0,scale])

def plot_M_distance(dist_train,scale=5):
    plt.figure()
    sns.distplot(dist_train,
                bins = 34, 
                kde= True, 
                color = 'green')
    plt.xlim([0.0,4])
    plt.xlabel('Mahalanobis dist')

def plot_train(dataset_train, scale_1=12, scale_2=6):
    dataset_train.plot(figsize = (scale_1,scale_2))

def plot_df(df, window=2):
    f,ax=plt.subplots(figsize=(7,3))
    df.rolling(window=window,center=True).median().plot(ax=ax)

def resample_reformat_df(df, resample_time='20T', is_time_index=False):
    if not is_time_index:
        df = df.set_index('Time')
    df = df.resample(resample_time).mean().dropna()
    df = df.reset_index().dropna()
    return df

def define_train_test(df, splits, resample, resample_time):
    from datetime import datetime
    if isinstance(splits[1], int):
        dataset_train = df[:splits[1]]
        dataset_test = df[splits[1]:]
    else:
        d = datetime.strptime(splits[1], "%d/%m/%Y")
        dataset_train = df.set_index('Time')[:d]
        dataset_test = df.set_index('Time')[d:]
        dataset_train = dataset_train.reset_index()
        dataset_test = dataset_test.reset_index()
        print(f"Train starts at {dataset_train.iloc[0]['Time']} and ends at {dataset_train.iloc[-1]['Time']}")
        print(f"Test starts at {dataset_test.iloc[0]['Time']} and ends at {dataset_test.iloc[-1]['Time']}")
    
    #dataset_train.pop('Time')
    #dataset_test.pop('Time')
    dataset_train = dataset_train.set_index('Time')
    dataset_test = dataset_test.set_index('Time')

    dataset_train = dataset_train.dropna()
    dataset_test = dataset_test.dropna()

    return dataset_train, dataset_test

def find_index_by_substring(df, substring):
    a = list(map(lambda x: x.find(substring), df['Time']))
    #a = a.index(0)
    b = next((i for i, x in enumerate(a) if x != -1), None)
    print(b)
    return b

def read_csv_transform_datetime(path,splits, sep, nrows=None, year=None, resample = False, resample_time = '5T'):
    if nrows is not None:
        df = pd.read_csv(path, sep=sep, nrows=nrows)
    else:
        df = pd.read_csv(path, sep=sep)
    
    index_0 = splits[0]
    index_2 = splits[2]
    if isinstance(splits[0], str):
        index_0 = find_index_by_substring(df, splits[0])
    if isinstance(splits[2], str):
        index_2 = find_index_by_substring(df, splits[2])
    
    df = df[index_0:index_2]

    datetime = pd.to_datetime(df['Time'], format=date_format)
    df['Time'] = datetime
    if year is not None:
        df = df[df['Time'].dt.year == year]
    return df

def init_datasets(path_1, path_2, splits, year=None,sep=';', nrows=None, resample=False, resample_time='5T'):
    df_1 = read_csv_transform_datetime(path_1, splits, sep=sep, nrows=nrows,year=year, resample=resample, resample_time=resample_time)
    df_2 = read_csv_transform_datetime(path_2, splits, sep=sep,nrows=nrows,year=year, resample=resample, resample_time=resample_time)
    df = pd.merge_asof(df_1.sort_values('Time'),
                       df_2.sort_values('Time'),
                       on='Time')
    if resample:
        df = resample_reformat_df(df, resample_time)

    dataset_train, dataset_test = define_train_test(df, splits, resample=resample, resample_time=resample_time)
    return dataset_train, dataset_test


def init_datasets_prepared(df_list, splits, year=None,sep=';', nrows=None, resample=False, resample_time='5T'):
    if len(df_list) == 2:
        df = pd.merge_asof(df_list[0].sort_values('Time'),
                        df_list[1].sort_values('Time'),
                        on='Time')
    elif len(df_list) > 2:
        df_rest = df_list[2:]
        df = pd.merge_asof(df_list[0].sort_values('Time'),
                        df_list[1].sort_values('Time'),
                        on='Time')
        for dframe in df_rest:
            df = pd.merge_asof(df.sort_values('Time'),
                            dframe.sort_values('Time'),
                            on='Time')

    if resample:
        df = resample_reformat_df(df, resample_time)

    dataset_train, dataset_test = define_train_test(df, splits, resample=resample, resample_time=resample_time)
    return dataset_train, dataset_test


def merge_asof_multiple(df_list, splits, year=None,sep=';', nrows=None):
    if len(df_list) == 2:
        df = pd.merge_asof(df_list[0].sort_values('Time'),
                        df_list[1].sort_values('Time'),
                        on='Time')
    elif len(df_list) > 2:
        df_rest = df_list[2:]
        df = pd.merge_asof(df_list[0].sort_values('Time'),
                        df_list[1].sort_values('Time'),
                        on='Time')
        for dframe in df_rest:
            df = pd.merge_asof(df.sort_values('Time'),
                            dframe.sort_values('Time'),
                            on='Time')
    
    return df
