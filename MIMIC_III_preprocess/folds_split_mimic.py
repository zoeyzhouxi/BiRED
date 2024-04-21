import pandas as pd
import numpy as np
import ujson as json

complete_tensor = pd.read_csv("./Data/Clean_data/complete_tensor.csv")
# complete_tensor = pd.read_csv(outfile_path + "segmented_tensor_train.csv")
complete_tensor.insert(0, 'hour', complete_tensor['TIME_STAMP']/60)
complete_tensor['hour'] = complete_tensor['hour'].apply(lambda x: round(x))

train_num = int(21260 * 0.8 + 1)

def parse_delta(masks, dir_, feature_no):
    if dir_ == 'backward':
        masks = masks[::-1]

    deltas = []

    for h in range(48):
        if h == 0:
            deltas.append(np.ones(feature_no))
        else:
            deltas.append(np.ones(feature_no) + (1 - masks[h]) * deltas[-1])

    return np.array(deltas)

def parse_rec(values, masks, evals, eval_masks, stamps, dir_, feature_no):
    deltas = parse_delta(masks, dir_, feature_no)

    # only used in GRU-D
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).values

    rec = {}
    rec['values'] = np.nan_to_num(values).tolist()
    rec['masks'] = masks.astype('int32').tolist()
    # rec['X_last_obsv'] = np.nan_to_num(X_last_obsv).tolist()
    # rec['x_mean'] = x_mean.tolist()
    # imputation ground-truth
    rec['evals'] = np.nan_to_num(evals).tolist()
    # rec['original_masks'] = original_masks.astype('int32').tolist()
    rec['eval_masks'] = eval_masks.astype('int32').tolist()
    rec['forwards'] = forwards.tolist()
    rec['deltas'] = deltas.tolist()
    rec['stamps'] = stamps.tolist()

    return rec


def mimic(set, missing_rate):
    if set=='train':
        fs = open('./Data/{}'.format(set), 'w')
        start = 0
        end = train_num
    else:
        fs = open('./Data/{}{}'.format(set, missing_rate), 'w')
        start = train_num
        end = 21261
    for id in range(start, end):
        out = pd.read_csv("./Data/Clean_data/complete_death_tags.csv").set_index('UNIQUE_ID')['Value']
        datafame = complete_tensor.loc[complete_tensor['UNIQUE_ID'] == id]

        stamps = []
        evals = []
        for h in range(48):
            x = datafame[datafame['hour']==h]
            x_ = x.set_index('LABEL_CODE').to_dict()['VALUENORM']
            t_ = x.set_index('LABEL_CODE').to_dict()['TIME_STAMP']
            value = []
            stamp = []
            for attr in range(96):
                if attr in x_:
                    # print(attr)
                    value.append(x_[attr])
                    stamp.append(t_[attr])
                else:
                    value.append(np.nan)
                    stamp.append(h * 60)
            stamps.append(stamp)
            evals.append(value)

        evals = np.array(evals)
        stamps = np.array(stamps)
        # data = pd.DataFrame(evals)
        # data.to_csv("./Data/test/{}.csv".format(id))  # Test data

        shp = evals.shape

        evals = evals.reshape(-1)
        values = evals.copy()
        if set=='test':
            indices = np.where(~np.isnan(evals))
            indices = np.array(indices).reshape(-1).tolist()
            indices = np.random.choice(indices, (len(indices) * missing_rate) // 100)
            values[indices] = np.nan

        masks = ~np.isnan(values)
        eval_masks = (~np.isnan(values)) ^ (~np.isnan(evals))

        evals = evals.reshape(shp)
        values = values.reshape(shp)

        masks = masks.reshape(shp)
        eval_masks = eval_masks.reshape(shp)

        label = out.loc[int(id)].tolist()  # 时间序列预测分类task标签
        rec = {'label': label}
        # prepare the model for both directions
        rec['forward'] = parse_rec(values, masks, evals, eval_masks, stamps, 'forward', 96)
        rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], stamps[::-1], 'backward', 96)
        # print(rec)
        rec = json.dumps(rec)
        fs.write(rec + '\n')

    fs.close()


mimic('test', 50)