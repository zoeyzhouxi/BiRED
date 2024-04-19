import torch
from torch import optim
from torch.utils.data import Subset
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, mean_absolute_error
import data_loader
from model import bired
import utils
import os
current_path = os.path.dirname(__file__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_encoder(encoder, dataset_name, learning_rate, rnn_type):

    train_dataset = data_loader.get_dataset(current_path+'/Data/json/{}/train'.format(dataset_name))
    val_dataset = data_loader.get_dataset(current_path+'/Data/json/{}/test10'.format(dataset_name))

    train_iter = data_loader.get_loader(batch_size=64, dataset=train_dataset, shuffle = True)
    val_iter = data_loader.get_loader(batch_size=64, dataset=val_dataset, shuffle = False)
    
    optimizer = optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=1e-5)
    min_loss = 30.0
    patience = 50
    for epoch in range(1000):

        '''-----------------------train---------------------'''
        encoder.train()
        sum_loss = 0
        for idx, data in enumerate(train_iter):
            optimizer.zero_grad()
            data = utils.to_var(data)
            if rnn_type=='LSTM':
                _, _, loss = encoder(data, 'forward')  # [batch,seq]
            else:
                _, loss = encoder(data, 'forward')
            sum_loss += loss

            loss.backward()
            optimizer.step()

            if (idx + 1) == len(train_iter):
                print("Progress_epoch %d ========>> average_loss:%.8f" % (epoch + 1, sum_loss / (idx + 1)))

        '''----------------------evaluate-----------------------'''
        encoder.eval()
        sum_loss = 0.0
        for idx, data in enumerate(val_iter):
            data = utils.to_var(data)
            if rnn_type=='LSTM':
                _, _, loss = encoder(data, 'forward')  # [batch,seq]
            else:
                _, loss = encoder(data, 'forward')
            sum_loss += loss

           
        print('evaluate_loss: {:.6f}'.format(sum_loss/(idx+1)))

        if ((sum_loss/(idx+1)) < min_loss):
            min_loss = sum_loss/(idx+1)
            torch.save(encoder.state_dict(),
                        current_path+'/results/Autoencoder/{}/encoder.pkl'.format(dataset_name))  
            patience_counter = 0
        else:
            patience_counter += 1 
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

def pre_train(dataset_name, hidden_size, output_size, feature_dim, seq_len, model_name, task_weight):

    if dataset_name=='PhysioNet':
        encoder = bired.Encoder(hidden_size, feature_dim, seq_len, 'GRU')
        encoder.load_state_dict(torch.load(current_path+'/results/Autoencoder/{}/encoder.pkl'.format(dataset_name)))
        model = bired.Model(hidden_size, feature_dim, seq_len, output_size, 1.0, 'GRU').to(device)
    else:
        encoder = bired.Encoder(hidden_size, feature_dim, seq_len, 'LSTM')
        encoder.load_state_dict(torch.load(current_path+'/results/Autoencoder/{}/encoder.pkl'.format(dataset_name)))
        model = bired.Model(hidden_size, feature_dim, seq_len, output_size, 1.0, 'LSTM').to(device)

    if torch.cuda.is_available():
        model = model.cuda()

    train_dataset = data_loader.get_dataset(current_path+'/Data/json/{}/train'.format(dataset_name))
    val_dataset = data_loader.get_dataset(current_path+'/Data/json/{}/test10'.format(dataset_name))

    train_iter = data_loader.get_loader(batch_size=64, dataset=train_dataset, shuffle = True)
    val_iter = data_loader.get_loader(batch_size=64, dataset=val_dataset, shuffle = False)

    min_MAE = 1
    max_AUC = 0.50
    patience = 20
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
   
    print('Start Training')
    for epoch in range(1000):   
        model.train()
        
        for idx, data in enumerate(train_iter):
           
            optimizer.zero_grad()
           
            data = utils.to_var(data)
            results = model(data=data)  # [batch,seq]
            loss = results['loss']

            loss.backward()
            optimizer.step()
            
        '''----------------------evaluate-----------------------'''
        model.eval()
        imputation = []
        eval = []
        eval_mask = []
        prediction = []
        label = []
        with torch.no_grad():
            for idx, data in enumerate(val_iter):
                data = utils.to_var(data)
                result = model(data)
                imputation.append(result['imputations'])
                eval.append(result['evals'])
                eval_mask.append(result['eval_masks'])
                prediction.append(result['predictions'])
                label.append(result['labels'])
            imputations = torch.cat(imputation, dim=0)
            imputations = imputations.data.cpu().numpy()  # [2396,seq,feature]
            evals = torch.cat(eval, dim=0)
            evals = evals.data.cpu().numpy()  # [2396,seq]
            eval_masks = torch.cat(eval_mask, dim=0)
            eval_masks = eval_masks.data.cpu().numpy()  # [2396,seq]
            predictions = torch.cat(prediction, dim=0)
            predictions = predictions.data.cpu().numpy()
            labels = torch.cat(label, dim=0)
            labels = labels.data.cpu().numpy()
            
            evals = np.asarray(evals[np.where(eval_masks == 1)].tolist())
            imputations = np.asarray(imputations[np.where(eval_masks == 1)].tolist())
                    
            predictions = np.asarray(predictions.tolist())
            labels = np.asarray(labels.tolist()).astype('int32')
        
            AUC = roc_auc_score(labels, predictions)
            MAE = mean_absolute_error(evals, imputations)
    
        if ((MAE < min_MAE) & (AUC > max_AUC)):
            print(f'Epoch {epoch + 1}, MAE: {MAE}, AUC: {AUC}%')
            torch.save(model.state_dict(),
                        current_path+'/results/model/{}/{}_{}.pkl'.format(dataset_name, model_name, task_weight))
            min_MAE = MAE
            max_AUC = AUC

            patience_counter = 0
        else:
            patience_counter += 1 
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break


def train(epoches, dataset_name, hidden_size, feature_dim, seq_len, output_size, model_name, task_weight):
    # 设置交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    train_dataset = data_loader.get_dataset(filenames='/Data/{}/train'.format(dataset_name))

    if dataset_name=='PhysioNet':
        model = bired.Model(hidden_size, feature_dim, seq_len, output_size, task_weight, 'GRU').to(device)
        model.load_state_dict(torch.load(current_path+'/results/model/{}/{}_1.0.pkl'.format(dataset_name, model_name)))
    else:
        model = bired.Model(hidden_size, feature_dim, seq_len, output_size, task_weight, 'LSTM').to(device)
        model.load_state_dict(torch.load(current_path+'/results/model/{}/{}_1.0.pkl'.format(dataset_name, model_name)))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))

    for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
        print(f'Fold {fold + 1}')
        
        # 创建数据子集
        train_subsampler = Subset(train_dataset, train_idx)
        val_subsampler = Subset(train_dataset, val_idx)
        
        # 创建数据加载器
        train_loader = data_loader.get_loader(batch_size=64, dataset=train_subsampler, shuffle=True)
        val_loader = data_loader.get_loader(batch_size=64, dataset=val_subsampler, shuffle=False)
        
        # 优化器
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        patience = 20

        # 训练模型
        for epoch in range(epoches):
            model.train()
            for idx, data in enumerate(train_loader):

                data = utils.to_var(data)
                optimizer.zero_grad()
                results = model(data=data)  # [batch,seq]
                loss = results['loss']
                loss.backward()
                optimizer.step()
            
            # 验证模型
            model.eval()
            with torch.no_grad():
                imputation = []
                evals = []
                eval_mask = []
                prediction = []
                label = []
                for idx, data in enumerate(val_loader):
                    data = utils.to_var(data)
                    result = model(data)
                    imputation.append(result['imputations'])
                    evals.append(result['evals'])
                    eval_mask.append(result['eval_masks'])
                    prediction.append(result['predictions'])
                    label.append(result['labels'])
                imputations = torch.cat(imputation, dim=0)
                imputations = imputations.data.cpu().numpy()  # [2396,seq,feature]
                evals = torch.cat(eval, dim=0)
                evals = evals.data.cpu().numpy()  # [2396,seq]
                eval_masks = torch.cat(eval_mask, dim=0)
                eval_masks = eval_masks.data.cpu().numpy()  # [2396,seq]
                predictions = torch.cat(prediction, dim=0)
                predictions = predictions.data.cpu().numpy()
                labels = torch.cat(label, dim=0)
                labels = labels.data.cpu().numpy()

                evals = np.asarray(evals[np.where(eval_masks == 1)].tolist())
                imputations = np.asarray(imputations[np.where(eval_masks == 1)].tolist())
                        
                predictions = np.asarray(predictions.tolist())
                labels = np.asarray(labels.tolist()).astype('int32')
            
                AUC = roc_auc_score(labels, predictions)
                MAE = mean_absolute_error(evals, imputations)

            if ((MAE < min_MAE) & (AUC > max_AUC)):
                print(f'Epoch {epoch + 1}, MAE: {MAE}, AUC: {AUC}%')
                torch.save(model.state_dict(),
                            current_path+'/results/model/{}/{}_{}.pkl'.format(dataset_name, model_name, task_weight))
                min_MAE = MAE
                max_AUC = AUC

                patience_counter = 0
            else:
                patience_counter += 1 
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

if __name__ == '__main__':
    # train(epoches=1000, dataset_name='PhysioNet', hidden_size=64, feature_dim=35, seq_len=48, output_size=1, model_name='bired', task_weight=0.4)
    train(epoches=1000, dataset_name='mimic', hidden_size=256, feature_dim=96, seq_len=48, output_size=1, model_name='bired', task_weight=0.4)