# from _future_ import absolute_import, division, print_function
#%load_ext autoreload
#%autoreload 2

import sys, os

from sklearn.preprocessing import StandardScaler

#from rrs_kit.DataClass import DataPath

script_dir  = os.path.normpath(os.path.abspath("."))
root_dir    = os.path.normpath(os.path.abspath(script_dir + "/../../../.."))
source_dir  = os.path.normpath(os.path.abspath(script_dir + "/../.."))
if source_dir in sys.path: sys.path.remove(source_dir)
sys.path.insert(1, source_dir)

import warnings
warnings.filterwarnings("ignore")
from common import *

dp = DataPath()

def preprocc_train(data_path, window_len, save_dir):
    df_in = pd.read_hdf(data_path, key = 'data')
    df_in = df_in.fillna(df_in.median())

    features_list = ['Albumin', 'Hgb', 'BUN', 'Alkaline phosphatase', 'WBC Count',
                     'SBP', 'Gender', 'Total calcium', 'RR', 'Age', 'Total bilirubin',
                     'Creatinin', 'ALT', 'Lactate', 'SaO2', 'AST', 'Glucose', 'Sodium', 'BT',
                     'HR', 'CRP', 'Chloride', 'Potassium', 'platelet', 'Total protein']

    df_train_data = df_in[features_list + ['Patient']]
    df_data = df_train_data.copy()
    #window_len = 16
    scaler = StandardScaler()

    if scaler is not None:
        scaler = StandardScaler()
        scaler.fit(df_train_data[features_list])
        features = scaler.transform(df_train_data[features_list])
        for idx, feature_name in enumerate(features_list):
            df_data[feature_name] = features[:, idx]

    patient_cnts = np.unique(df_data["Patient"], return_counts=True)
    patient_cnts = dict(zip(patient_cnts[0], patient_cnts[1]))
    patient_ids = list(patient_cnts.keys())

    patient_abnormal_ids = np.unique(df_data.query('target==1')['Patient'])
    patient_normal_ids = np.unique(df_data.query('target==0')['Patient'])

    data_info = {}
    for patient_id in tqdm.tqdm(patient_ids):
        df_patient = df_data.query(f'Patient=={patient_id}')
        for idx in range(len(df_patient) - window_len + 1):
            row_info = {}
            from_idx = idx
            to_idx = idx + window_len - 1

            ############# row ################
            row_info["pid"] = df_patient["Patient"].iloc[from_idx: to_idx + 1].values
            row_info["x"] = df_patient[features_list].iloc[from_idx: to_idx + 1].values
            row_info["y"] = df_patient["target"].iloc[from_idx: to_idx + 1].values
            row_info["seq_y"] = df_patient["target"].iloc[to_idx]

            ############ append ##############
            for key in row_info:
                if data_info.get(key) is None: data_info[key] = []
                data_info[key].append(row_info[key])
                pass  # key

            # print(f'{idx} - {idx + window_len - 1}')
            # break
            pass  # row
        pass  # data

    for key in row_info:
        data_info[key] = np.array(data_info[key])
        pass  # key

    np.savez(f'{save_dir}/data_info.npz', **data_info)
    joblib.dump(scaler, f'{save_dir}/scaler.joblib')
    df_data.to_hdf(f'{save_dir}/df_data.hdf', key='data')

    save_info = {
        "patient_ids": patient_ids,
        "patient_cnts": patient_cnts,
        "window_len": window_len,
        "patient_abnormal_ids": patient_abnormal_ids,
        "patient_normal_ids": patient_normal_ids,
        "features_list": features_list,
        "all_column_list": list(df_data.columns),
    }

    joblib.dump(save_info, f'{save_dir}/save_info.joblib')
    return data_info

def preprocc_test(data_path, window_len=8):
    df_test = pd.read_csv(data_path) # or read_hdf
    scaler = StandardScaler()
    features_list = [x for x in df_test.columns.tolist() if x not in ['Patient']]

    if scaler is not None:
        scaler = StandardScaler()
        scaler.fit(df_test[features_list])
        features = scaler.transform(df_test[features_list])
        for idx, feature_name in enumerate(features_list):
            df_test[feature_name] = features[:, idx]

    patient_cnts = np.unique(df_test["Patient"], return_counts=True)
    patient_cnts = dict(zip(patient_cnts[0], patient_cnts[1]))
    patient_ids = list(patient_cnts.keys())

    data_info = {}
    for patient_id in tqdm.tqdm(patient_ids):
        df_patient = df_test.query(f'Patient=={patient_id}')
        for idx in range(len(df_patient) - window_len + 1):
            row_info = {}
            from_idx = idx
            to_idx = idx + window_len - 1

            ############# row ################
            row_info["pid"] = df_patient["Patient"].iloc[from_idx: to_idx + 1].values
            row_info["x"] = df_patient[features_list].iloc[from_idx: to_idx + 1].values

            ############ append ##############
            for key in row_info:
                if data_info.get(key) is None: data_info[key] = []
                data_info[key].append(row_info[key])
                pass  # key

            pass  # row
        pass  # data

    for key in row_info:
        data_info[key] = np.array(data_info[key])
        pass

    return data_info