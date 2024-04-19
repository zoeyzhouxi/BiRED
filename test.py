import torch
import numpy as np
import time
import utils
from model import bired
import data_loader
from sklearn.metrics import mean_squared_error, roc_auc_score, mean_absolute_error, roc_curve, auc
import os
current_path = os.path.dirname(__file__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(dataset_name, model_name, missing_rate, hidden_size, feature_dim, seq_len, output_size, task_weight):
    print('{}_dataset'.format(dataset_name))
    print(model_name)
    if dataset_name=='PhysioNet':
        model = bired.Model(hidden_size, feature_dim, seq_len, output_size, task_weight, 'GRU').to(device)
    else:
        model = bired.Model(hidden_size, feature_dim, seq_len, output_size, task_weight, 'LSTM').to(device)

    test_dataset = data_loader.get_dataset(current_path+'/Data/json/%s/test%d' % (dataset_name, missing_rate))

    test_iter = data_loader.get_loader(batch_size=64, dataset=test_dataset, shuffle = False)
    
    
    model.load_state_dict(torch.load(current_path+'/results/model/{}/{}_{}.pkl'.format(dataset_name, model_name, task_weight)))
    
    if torch.cuda.is_available():
        model = model.cuda()
        
    model.eval()
    imputation = []
    eval = []
    eval_mask = []
    prediction = []
    label = []
    with torch.no_grad():
        torch.cuda.synchronize()
        start = time.time()
        for idx, data in enumerate(test_iter):
            data = utils.to_var(data)
            if model_name == 'SSIM':
                result = model(data, 0.5)
            else:
                result = model(data=data)  # [batch,seq]
                prediction.append(result['predictions'])
                label.append(result['labels'])
            imputation.append(result['imputations'])
            eval.append(result['evals'])
            eval_mask.append(result['eval_masks'])
        torch.cuda.synchronize()
        print(time.time()-start)
        imputations = torch.cat(imputation, dim=0)
        imputations = imputations.data.cpu().numpy()  # [2396,seq,feature]
        evals = torch.cat(eval, dim=0)
        evals = evals.data.cpu().numpy()  # [2396,seq]
        eval_masks = torch.cat(eval_mask, dim=0)
        eval_masks = eval_masks.data.cpu().numpy()  # [2396,seq]
    
        evals = np.asarray(evals[np.where(eval_masks == 1)].tolist())
        imputations = np.asarray(imputations[np.where(eval_masks == 1)].tolist())
        MSE = mean_squared_error(evals, imputations)
        MAE = mean_absolute_error(evals, imputations)
        RMSE = np.sqrt(MSE)
        MRE = sum(np.abs(evals - imputations)) / sum(np.abs(evals))
        
       
        predictions = torch.cat(prediction, dim=0)
        predictions = predictions.data.cpu().numpy()
        labels = torch.cat(label, dim=0)
        labels = labels.data.cpu().numpy()
        predictions = np.asarray(predictions.tolist())
        labels = np.asarray(labels.tolist()).astype('int32')
        AUC = roc_auc_score(labels, predictions)
        fpr, tpr, thresholds = roc_curve(labels, predictions)
       
        return MAE, MRE, RMSE, MSE, AUC, fpr, tpr
        

def show(dataset_name, model_name, feature_dim, hidden_size, output_size, task_weight, seq_len):
    
    results = np.zeros([5, 9])
    ls = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    for i in range(1): 
        MAE1, MRE1, RMSE1, MSE1, AUC1, fpr1, tpr1 = evaluate(dataset_name, model_name, ls[i], hidden_size, feature_dim, seq_len, output_size, task_weight)
        if i==8:
            np.save(current_path+'/results/result/{}/{}_fpr'.format(dataset_name, model_name), fpr1)
            np.save(current_path+'/results/result/{}/{}_tpr'.format(dataset_name, model_name), tpr1)
    
        print(ls[i], 'MAE:%.6f' % (MAE1), 'MRE:%.6f' % (MRE1), 'RMSE:%.6f' % (RMSE1), 'AUC:%.6f'%(AUC1))
        results[0, i] = MAE1
        results[1, i] = MRE1
        results[2, i] = RMSE1
        results[3, i] = MSE1
        results[4, i] = AUC1
    print(results)
    np.save(current_path+'/results/result/{}/{}'.format(dataset_name, model_name), results)

if __name__ == '__main__':
    # show(dataset_name='PhysioNet', model_name='BiRED', feature_dim=35, hidden_size=64, output_size=1, task_weight=0.4, seq_len=48)
    show(dataset_name='mimic-iii', model_name='BiRED', feature_dim=96, hidden_size=256, output_size=1, task_weight=0.4, seq_len=48)