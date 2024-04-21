import os
import re
import warnings
import numpy as np
import pandas as pd
import ujson as json
warnings.filterwarnings('ignore')
seq_len = 48

def to_time_bin(x):
    h, m = map(int, x.split(':'))
    t = h * 60 + m
    h = round(t/60)
    return h

def timestamps(x):
    h, m = map(int, x.split(':'))
    t = h * 60 + m
    return t

def parse_data(x, h, attributes):

    x_ = x.set_index('Parameter').to_dict()['Value']
    t_ = x.set_index('Parameter').to_dict()['timestamps']
    values = []
    stamps = []
    for attr in attributes:
        if attr in x_:
    
            t = t_[attr]
            values.append(x_[attr])
            stamps.append(t)
        else:
            values.append(np.nan)
            stamps.append(h*60)
    
    return values, stamps


def parse_delta(masks):
    feature_no = masks.shape[1]
    deltas = []

    for h in range(seq_len):
        if h == 0:
            deltas.append(np.zeros(feature_no))
        else:
            deltas.append(np.ones(feature_no) + (1 - masks[h]) * deltas[-1])
    deltas = np.array(deltas)
    # deltas = deltas/47.0
    return deltas


def parse_rec(values, masks, evals, eval_masks, stamps):
    deltas = parse_delta(masks)
    print(deltas)
    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).values

    rec = {}
    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    rec['evals'] = np.nan_to_num(evals).tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()
    rec['stamps'] = stamps.tolist()

    return rec


def parse_id(set, id_, attributes, mean, std, out, missing_rate):
    data = pd.read_csv('./Data/{}/{}.txt'.format(set, id_))
    t = data['Time']
    t = t.apply(lambda x: timestamps(x))
    data.insert(0, 'timestamps', t)
    # accumulate the records within one hour
    data['Time'] = data['Time'].apply(lambda x: to_time_bin(x))
   
    evals = []
    stamps = []
    # merge all the metrics within one hour
    for h in range(seq_len):
        v, t = parse_data(data[data['Time'] == h], h, attributes)
        stamps.append(t)
        evals.append(v)

    evals = (np.array(evals) - mean) / std
    evals = np.array(evals)
    stamps = np.array(stamps)
    shp = evals.shape

    evals = evals.reshape(-1)
    values = evals.copy()
    if set == 'test':
    # if set == 'train':
        # randomly eliminate 20% values as the imputation ground-truth
        indices = np.where(~np.isnan(evals))[0].tolist()
        indices = np.random.choice(indices, (len(indices)*missing_rate) // 100)
        values[indices] = np.nan

    masks = ~np.isnan(values)
    eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))

    evals = evals.reshape(shp)
    values = values.reshape(shp)
    masks = masks.reshape(shp)
    eval_masks = eval_masks.reshape(shp)

    '''{标签：，前向：{}，后向：{}}，写入json文件'''
    label = out.loc[int(id_)].tolist()  # 时间序列预测分类task标签
    rec = {'label': label}
    # prepare the model for both directions
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, stamps)
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], stamps[::-1])
    # print(rec)
    rec = json.dumps(rec)
    return rec


def PhysioNet(set, missing_rate):
    attributes = ['DiasABP', 'HR', 'NIDiasABP', 'RespRate', 'SysABP', 'FiO2',
                  'NISysABP', 'MAP', 'Urine', 'NIMAP', 'Temp', 'GCS',
                  'Na', 'Lactate', 'PaO2', 'WBC', 'pH', 'Albumin',
                  'ALT', 'Glucose', 'SaO2', 'AST', 'Bilirubin', 'HCO3',
                  'BUN', 'Mg', 'HCT', 'K', 'Cholesterol', 'TroponinT',
                  'TroponinI', 'PaCO2', 'Platelets', 'Creatinine', 'ALP']
    mean = [59.540976152469405, 86.72320413227443, 58.13833409690321, 19.64795551193981, 119.60137167841977,
            0.5404785381886381,
            119.15012244292181, 80.20321011673151, 116.1171573535279, 77.08923183026529, 37.07362841054398,
            11.407767149315339,
            139.06972964987443, 2.8797765291788986, 147.4835678885565, 12.670222585415166, 7.490957887101613,
            2.922874149659863,
            394.8899400819931, 141.4867570064675, 96.66380228136883, 505.5576196473552, 2.906465787821709,
            23.118951553526724,
            27.413004968675743, 2.0277491155660416, 30.692432164676188, 4.135790642787733, 156.51746031746032,
            1.2004983498349853,
            7.127188940092161, 40.39875518672199, 191.05877024038804, 1.5052390166989214, 116.77122488658458]
    std = [13.01436781437145, 17.789923096504985, 15.06074282896952, 5.50914416318306, 23.730556355204214,
           0.18634432509312762,
           21.97610723063014, 16.232515568438338, 170.65318497610315, 14.856134327604906, 1.5604879744921516,
           3.967579823394297,
           5.185595006246348, 2.5287518090506755, 85.96290370390257, 7.649058756791069, 8.384743923130074,
           0.6515057685658769,
           1201.033856726966, 67.62249645388543, 3.294112002091972, 1515.362517984297, 5.902070316876287,
           4.707600932877377,
           23.403743427107095, 0.4220051299992514, 5.002058959758486, 0.706337033602292, 45.99491531484596,
           22.716532297586456,
           9.754483687298688, 9.062327978713556, 106.50939503021543, 1.6369529387005546, 133.96778334724377]

    patient_ids = []
    for filename in os.listdir('./Data/{}'.format(set)):
        # the patient data in PhysioNet contains 6-digits
        match = re.search('\d{6}', filename)
        if match:
            id_ = match.group()
            patient_ids.append(id_)

    out = pd.read_csv('./Data/Outcomes.txt').set_index('RecordID')['In-hospital_death']

    if set=='train':
        fs = open('./Data/json/PhysioNet/{}'.format(set), 'w')
    else:
        fs = open('./Data/json/PhysioNet/{}{}'.format(set, missing_rate), 'w')
    for id_ in patient_ids:
        print('Processing patient {}'.format(id_))
        try:
            rec = parse_id(set, id_, attributes, mean, std, out, missing_rate)
            # fs.write(rec + '\n')
        except Exception as e:
            print(e)
            continue

    # fs.close()

# PhysioNet('train', 10)
PhysioNet('test', 10)