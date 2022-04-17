import os.path

from common import *
from models import *
from CNUH_preprocess import *
import warnings
warnings.filterwarnings("ignore")

data_path = 'data_path'
model_path = 'weight_path'


def get_prediction_first(input_path, model_name):
    print('Pre-process data ..')
    def load_input_data(input_path):
        df_preprocc = preprocc_test(input_path)
        in_features = df_preprocc['x']
        #print(in_features.shape)
        return in_features

    X_test = load_input_data(input_path)

    if model_name == 'tvae':
        load_model_path = f'{model_path}/model_VAE_KFold_1.hdf5'
        model = build_VAE(input_shape=X_test.shape[1:], latent_dim=8)
        model.load_weights(load_model_path)

        rs_prob = model.predict(X_test)
        rs_label = np.argmax(rs_prob, axis=1)

    if model_name == 'rnn':   # doi: 10.1161/JAHA.118.008678.
        model = build_kwon_RNN(input_shape=X_test.shape[1:])
        load_model_path = f'{model_path}/model_kwonRNN_KFold_1.hdf5'
        model = build_VAE(input_shape=X_test.shape[1:], latent_dim=8)
        model.load_weights(load_model_path)

        rs_prob = model.predict(X_test)
        rs_label = np.argmax(rs_prob, axis=1)

    if model_name == 'biLSTMATT': #DOI: 10.1109/JBHI.2019.2937803
        model == build_shamount_Att_BiLSTM(input_shape=X_test.shape[1:])
        load_model_path = f'{model_path}/model_BiLSTM_KFold_1.hdf5'
        model = build_VAE(input_shape=X_test.shape[1:], latent_dim=8)
        model.load_weights(load_model_path)

        rs_prob = model.predict(X_test)
        rs_label = np.argmax(rs_prob, axis=1)

    print('Abnormal probability: ', rs_prob[0][1])
    print('Abnormal(1)/normal(0) results: ', rs_label)


def get_next_hours_prediction(new_ts_path, curr_window_path, model_name, save_path):
    new_ts = pd.read_csv(new_ts_path)
    old_window = pd.read_csv(curr_window_path)

    new_window = old_window.loc[1:, :]
    new_window = new_window.append(new_ts, ignore_index = True)

    new_path = os.path.join(save_path,'new_window.csv')
    new_window.to_csv(new_path, index=False)
    return get_prediction_first(new_path, model_name)



#predict
#get_prediction_first('data_path/Input_data_sample.csv', 'tvae')
get_next_hours_prediction('data_path/Input_data_sample.csv', 'data_path/Sample_patient_measurement.csv', 'tvae', save_path=data_path)