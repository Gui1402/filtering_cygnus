import json
import numpy as np
import itertools
import pandas as pd

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx, array[idx]

def load_data(path):
    with open(path, 'r') as JSON:
        json_dict = json.load(JSON)
    return json.loads(json_dict)   



def name_change(name):
    er = name.split('ER')
    he = name.split('He')
    if len(er) > 1:
        return ['ER', er[-1].split('_')[1]]
    elif len(he) > 1:
        return ['He', he[-1].split('_')[1]]
    else:
        return None
      

def fill_nan_nn(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask,np.arange(mask.shape[1]),0)
    np.maximum.accumulate(idx,axis=1, out=idx)
    out = arr[np.arange(idx.shape[0])[:,None], idx]
    return out


def data_struct(input_list):
    array = np.array(input_list)
    out_array_list = []
    for n_image in range(array.shape[0]):
        for n_filter in range(array.shape[2]):
            out_array_list.append(list(array[n_image, :, n_filter]))
    return out_array_list


        
def tab_data(json_file):
    data = load_data(json_file)
    base = pd.DataFrame({'path':data['Image_path'], 'image':data['Image_index'], 'energia_ped': data['Energy']['image_real'], 'cluster_integral': data['Energy']['image_truth']})
    array_list = np.array(data['Energy']['image_after_threshold'])
    n_filters = array_list.shape[2]
    array_list = data_struct(array_list)
    roc_x = np.array(data['ROC']['full'])[:, 0, :, :]
    roc_y = np.array(data['ROC']['full'])[:, 1, :, :]
    threshold = np.array(data['ROC']['threshold'])[:,:,:,0,0] 
    roc_x_list = data_struct(roc_x)
    roc_y_list = data_struct(roc_y)
    threshold_list = data_struct(threshold)
    base = base.loc[base.index.repeat(n_filters)].reset_index(drop=True)
    base['filter'] = data['Filter_name']
    base['parameter'] = data['Filter_parameter']
    array_df = pd.DataFrame({'energy_after_threshold':array_list, 'recall': roc_x_list, 'precision': roc_y_list, 'threshold_all': threshold_list})
    array_df = array_df.reset_index(drop=True)
    return base.join(array_df)

def dataframe_column_adj(dataframe):
    df = dataframe.copy()
    seg_df = pd.DataFrame.from_records(df.path.apply(name_change).values, columns=['particle', 'energy'])
    seg_df.energy = seg_df.energy.astype(np.int)
    df['parameter'] = df.parameter.apply(lambda x: x[0])
    df.parameter[df.parameter == 'n'] = 0
    df.parameter[df.parameter.isnull()] = 0
    df.parameter = df.parameter.astype(np.int)
    df = seg_df.join(df.reset_index().drop('index', axis=1)).drop(['path'], axis=1)
    df['f1'] = df.apply(lambda x: max((2*np.array(x['recall'])*np.array(x['precision']))/(np.array(x['recall']) + np.array(x['precision']))), axis=1)
    return df