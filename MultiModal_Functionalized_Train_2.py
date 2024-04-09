import torch 
from torch_geometric.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader

import torch.nn as nn
import torch.optim as optim
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout, Sigmoid
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GCNConv,GATv2Conv,MessagePassing
import matplotlib.pyplot as plt
from torch.utils.data import random_split

import torch_geometric.nn as pyg_nn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.data import Batch, Data, Dataset, DataLoader

from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score,roc_curve, balanced_accuracy_score
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score, roc_curve, roc_auc_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

import pandas as pd

import numpy as np
from tqdm import tqdm
#import mlflow.pytorch
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd 
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Dataset, Data
import numpy as np 
import os
from Graph_Featurizer_coformer1 import MoleculeDataset_1
from Graph_Featurizer_coformer2 import MoleculeDataset_2
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import seaborn as sns
from molvs import standardize_smiles


from mordred import Calculator, descriptors
import deepchem as dc





class MPNNModel(MessagePassing, nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MPNNModel, self).__init__(aggr='mean')  # 'mean' aggregation for global pooling
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)

        self.fc = nn.Sequential(nn.Linear(hidden_dim * 2, 160),
                                  nn.BatchNorm1d(160),
                                  nn.ReLU(),
                                  nn.Dropout(),
                                  
                                  nn.Linear(160, 90),
                                  nn.BatchNorm1d(90),
                                  nn.ReLU(),
                                  nn.Linear(90, 1),
                                  nn.Sigmoid()
                                  )
            

    def forward(self, x, edge_index, batch):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)

        # Global Pooling (stack different aggregations)
        x = torch.cat([global_mean_pool(x, batch), 
                       global_max_pool(x, batch)], dim=1)

        x = self.fc(x)
        return x



class ANN_cls (nn.Module):
    def __init__(self, input_size, hid_size1, hid_size2, last_layer_size):
        super(ANN_cls,self).__init__()
        self.fc1 = nn.Linear(input_size, hid_size1)
        self.act_hid = nn.ReLU()
        self.fc2 = nn.Linear(hid_size1, hid_size2)
        self.fc3 = nn.Linear(hid_size2, last_layer_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act_hid(x)
        x = self.fc2(x)
        x = self.act_hid(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x





def combine_Graph(Graph_list):
    """
    merge a Graph with multiple subgraph
    Args:
        Graph_list: list() of torch_geometric.data.Data object

    Returns: torch_geometric.data.Data object

    """
    x = Batch.from_data_list(Graph_list).x
    edge_index = Batch.from_data_list(Graph_list).edge_index
    edge_attr = Batch.from_data_list(Graph_list).edge_attr

    combined_Graph = Data(x = x,edge_index = edge_index,edge_attr = edge_attr)

    return combined_Graph



def preprocess_graph_data(root_path_cof_1, filename_cof_1, root_path_cof_2, filename_cof_2):
    
    # Combine and process graph data
    dataset_graphs_1 = MoleculeDataset_1(root=root_path_cof_1, filename=filename_cof_1)
    dataset_graphs_2 = MoleculeDataset_2(root=root_path_cof_2, filename=filename_cof_2)
    combined_graphs = [combine_Graph([dataset_graphs_1[i], dataset_graphs_2[i]]) for i in range(len(dataset_graphs_1))]
    for i in range(len(dataset_graphs_1)):
        combined_graphs[i].y = dataset_graphs_2[i].y
        combined_graphs[i].smiles1 = dataset_graphs_1[i].smiles
        combined_graphs[i].smiles2 = dataset_graphs_2[i].smiles

    #Extract features and labels
    Labels = np.array([dataset_graphs_2[i].y.numpy() for i in range(len(dataset_graphs_1))])
    return combined_graphs, Labels

def Preprocess_Tabular_descriptors(root_path_cof_1, filename_cof_1, root_path_cof_2, filename_cof_2):
    
    smiles_cof_1 = pd.read_csv(os.path.join(root_path_cof_1, filename_cof_1))['smiles1']
    smiles_cof_2 = pd.read_csv(os.path.join(root_path_cof_2, filename_cof_2))['smiles2']
    
    calc = Calculator(descriptors, ignore_3D=False)
    Mordred_feature_names_1 = ['cof1_'+str(descriptor) for descriptor in calc.descriptors]
    Mordred_feature_names_2 = ['cof2_'+str(descriptor) for descriptor in calc.descriptors]
    
    featurizer = dc.feat.MordredDescriptors(ignore_3D=False)
    Mordred_smiles_1  = featurizer.featurize(smiles_cof_1)
    Mordred_smiles_2 = featurizer.featurize(smiles_cof_2)
    
    X = np.concatenate((Mordred_smiles_1, Mordred_smiles_2), axis = 1)
    X_df = pd.DataFrame(X, columns = Mordred_feature_names_1 + Mordred_feature_names_2)
    
    return X_df


def preprocess_combined_modalities(root_path_cof_1, filename_cof_1, root_path_cof_2, filename_cof_2, selected_features):
    # mordred features
    smiles_cof_1 = pd.read_csv(os.path.join(root_path_cof_1, filename_cof_1))['smiles1']
    smiles_cof_2 = pd.read_csv(os.path.join(root_path_cof_2, filename_cof_2))['smiles2']
    
    calc = Calculator(descriptors, ignore_3D=False)
    Mordred_feature_names_1 = ['cof1_'+str(descriptor) for descriptor in calc.descriptors]
    Mordred_feature_names_2 = ['cof2_'+str(descriptor) for descriptor in calc.descriptors]
    
    featurizer = dc.feat.MordredDescriptors(ignore_3D=False)
    Mordred_smiles_1  = featurizer.featurize(smiles_cof_1)
    Mordred_smiles_2 = featurizer.featurize(smiles_cof_2)
    X = np.concatenate((Mordred_smiles_1, Mordred_smiles_2), axis = 1)
    X_df = pd.DataFrame(X, columns = Mordred_feature_names_1 + Mordred_feature_names_2)
    X_df_selected = X_df[selected_features].values
    scaler = MinMaxScaler()
    X_df_selected_scaled = scaler.fit_transform(X_df_selected)
    
    # Combine and process graph data
    dataset_graphs_1 = MoleculeDataset_1(root=root_path_cof_1, filename=filename_cof_1)
    dataset_graphs_2 = MoleculeDataset_2(root=root_path_cof_2, filename=filename_cof_2)
    combined_graphs = [combine_Graph([dataset_graphs_1[i], dataset_graphs_2[i]]) for i in range(len(dataset_graphs_1))]
    
    for i in range(len(dataset_graphs_1)):
        combined_graphs[i].y = dataset_graphs_2[i].y
        combined_graphs[i].smiles1 = dataset_graphs_1[i].smiles
        combined_graphs[i].smiles2 = dataset_graphs_2[i].smiles
        combined_graphs[i].mordred = torch.tensor(X_df_selected_scaled[i,:].reshape(1,-1)).float()
        
    Mixed_graphs = combined_graphs

    #Extract features and labels
    Labels = np.array([dataset_graphs_2[i].y.numpy() for i in range(len(dataset_graphs_1))])
    return Mixed_graphs, Labels



"""Mordred Tabular Features"""

""" Mordred Descriptors"""
def Mordred_split_selection(X_df, Labels, Batch_size, selected_feat_no, test_size=0.2, val_size=0.1, random_state=104):
 
    y = Labels
    val_size = float(0.1)
    test_size = float(0.3)
    train_size = 1 - test_size - val_size
    
    # Split the dataset into training, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_df, y, test_size=test_size, random_state=random_state
    )
    
    # Further split the training set into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size/train_size, random_state=random_state
    )
    # Instantiate the MinMaxScaler
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_val_scaled = scaler.transform(X_val)
    
    # Train a Random Forest classifier for feature selection
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf_classifier.fit(X_train_scaled, y_train)
    
    # Get feature importances and select top features
    feature_importances = rf_classifier.feature_importances_
    feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    top_features = feature_importance_df.nlargest(selected_feat_no, 'Importance')
    selected_features = top_features['Feature'].tolist()
    
    #xgb_classifier = XGBClassifier(n_estimators=100, random_state=42)
    #xgb_classifier.fit(X_train_selected, y_train)
    #accuracy = xgb_classifier.score(X_test_selected, y_test)
    #print(f"XGBoost Classifier Accuracy: {accuracy}")
    #predictions = xgb_classifier.predict(X_test_selected)
    #print(classification_report(y_test, predictions))
    
    # Select the top features for training and validation sets
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    X_val_selected = X_val[selected_features]
    
    scaler = MinMaxScaler()
    X_train_selected_scaled = scaler.fit_transform(X_train_selected)
    X_test_selected_scaled = scaler.transform(X_test_selected)
    X_val_selected_scaled = scaler.transform(X_val_selected)
    
    X_train_tensor = torch.tensor (X_train_selected_scaled, dtype = torch.float32)
    X_test_tensor = torch.tensor (X_test_selected_scaled, dtype = torch.float32)
    X_val_tensor = torch.tensor(X_val_selected_scaled, dtype = torch.float32)

    y_train_tensor = torch.tensor (y_train, dtype = torch.float32)
    y_test_tensor = torch.tensor (y_test, dtype = torch.float32)
    y_val_tensor = torch.tensor (y_val, dtype = torch.float32)

    # Create TensorDataset for training and testing data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    # Create DataLoader objects for batches
    batch_size = Batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle test data
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)  # No need to shuffle test data


    return train_loader, test_loader, val_loader, selected_features




def graph_split_batch(dataset_graphs, Batch_size, train_size=0.7, val_size=0.1, random_state=104):
    
    NUM_GRAPHS_PER_BATCH = Batch_size  # batch size
    
    total_size = len(dataset_graphs)
    train_size = int(train_size * total_size)
    val_size = int(val_size * total_size)
    test_size = total_size - train_size - val_size
    
    random_state = 104
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset_graphs, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(random_state)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=NUM_GRAPHS_PER_BATCH, shuffle=False)
    print("Train DataLoader size:", len(train_loader))
    print("Validation DataLoader size:", len(val_loader))
    print("Test DataLoader size:", len(test_loader))
    
    return train_loader, val_loader, test_loader




def Train_Tabular(train_loader, val_loader, test_loader, num_epochs = 100, patience = 50):
    
    train_iter = iter(train_loader)
    X_train_sample, y_train_sample = next(train_iter)
    input_size = X_train_sample.shape[1]
    
    hid_size1 = 150
    hid_size2 = 100
    last_layer_size = 1
    
    mlp_model = ANN_cls(input_size, hid_size1, hid_size2, last_layer_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr = 0.001)

    
    early_stop_counter = 0
    best_val_loss = float('inf')
    best_model = None

    
    
    train_losses = []
    train_predictions = []
    train_targets = []
    train_accuracy, train_BACC, train_AUC = [], [], []
    
    val_losses = []
    val_accuracy, val_BACC, val_AUC = [], [], []
    
    for epoch in range(num_epochs):
        mlp_model.train()
        total_loss = 0.0
        
        for index, (X_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = mlp_model(X_batch)
            train_loss = criterion(outputs, y_batch)
         
            train_loss.backward()
            optimizer.step()
    
            total_loss += train_loss.item()*X_batch.size(0)  # Multiply by batch size to get total loss
            
            train_predictions.extend((outputs).detach().numpy().flatten())
            train_targets.extend(y_batch.view(-1).cpu().numpy())
    
        # Calculate average loss for the epoch
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Calculate accuracy for training data
        train_accuracy.append(accuracy_score(train_targets, np.round(train_predictions)))
        #print(train_accuracy)
        train_BACC.append(balanced_accuracy_score(train_targets, np.round(train_predictions)))
        train_AUC.append(roc_auc_score(train_targets, train_predictions))
    
        
        mlp_model.eval()
        
        total_loss = 0.0

        with torch.no_grad():
            val_predictions = []
            val_targets = []
            for index, (X_batch, y_batch) in enumerate(val_loader):
                outputs = mlp_model(X_batch)
                val_loss = criterion(outputs, y_batch.view(-1, 1).float())
             
                total_loss += val_loss.item()*X_batch.size(0)  # Multiply by batch size to get total loss
                
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(y_batch.view(-1).cpu().numpy())
           
            avg_val_loss = total_loss/len(val_loader.dataset)
            val_losses.append(avg_val_loss)
            val_predictions = np.concatenate(val_predictions)
            val_targets = np.concatenate(val_targets)
            epoch_val_acc = accuracy_score(val_targets, np.round(val_predictions))
            epoch_val_BACC = balanced_accuracy_score(val_targets, np.round(val_predictions))
            epoch_val_AUC = roc_auc_score(val_targets, val_predictions)
            val_accuracy.append(epoch_val_acc)
            val_BACC.append(epoch_val_BACC)
            val_AUC.append(epoch_val_AUC)
        
        
        # Update best model if the current validation RMSE is lower
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = mlp_model.state_dict()
            early_stop_counter = 0  # Reset the counter when a better model is found
        else:
            early_stop_counter += 1
    
        # Print and log to TensorBoard every 10 epochs
        if epoch % 10 == 0:

            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Balanced Accuracy: {epoch_val_BACC:.4f}, Validation AUC: {epoch_val_AUC:.4f}")
    
        # Early stopping check
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} as there is no improvement for {patience} consecutive epochs.")
            break
    
    
    
    
    # Evaluate on test data
        
    mlp_model.eval()
    test_losses = []
    test_predictions = []
    test_targets = []
    test_accuracy, test_BACC, test_AUC = [], [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for index, (X_batch, y_batch) in enumerate(test_loader):
            outputs = mlp_model(X_batch)
            test_loss = criterion(outputs, y_batch.view(-1, 1).float())
         
            total_loss += test_loss.item()*X_batch.size(0)  # Multiply by batch size to get total loss
            
            test_predictions.append(outputs.cpu().numpy())
            test_targets.append(y_batch.view(-1).cpu().numpy())
       
        avg_test_loss = total_loss/len(test_loader.dataset)
        test_predictions = np.concatenate(test_predictions)
        test_targets = np.concatenate(test_targets)

        test_accuracy = [accuracy_score(test_targets, np.round(test_predictions))]
        test_BACC = [balanced_accuracy_score(test_targets, np.round(test_predictions))]
        test_AUC = [roc_auc_score(test_targets, test_predictions)]
        test_recall = [recall_score(test_targets, np.round(test_predictions))]
        test_predictions = test_predictions.tolist()
        test_targets = test_targets.tolist()
        
        print(f"MLP Classifier Accuracy: {test_accuracy}")
        print(f"MLP Classifier Recall: {test_recall}")
        print(f"MLP Classifier Balanced Accuracy: {test_BACC}")
        print(f"MLP Classifier AUC: {test_AUC}")


    # Pad the shorter lists with None to make them equal in length
    max_length = max(len(train_losses), len(train_accuracy), len(train_BACC), len(train_AUC),
                  len(val_losses), len(val_accuracy), len(val_BACC), len(val_AUC),
                  len(test_predictions), len(test_targets), len(test_accuracy),
                  len(test_recall), len(test_BACC), len(test_AUC))

    def pad_list(lst, max_length, pad_value=None):
        return lst + [pad_value] * (max_length - len(lst))
    
    train_losses = pad_list(train_losses, max_length)
    train_accuracy = pad_list(train_accuracy, max_length)
    train_BACC = pad_list(train_BACC, max_length)
    train_AUC = pad_list(train_AUC, max_length)
    val_losses = pad_list(val_losses, max_length)
    val_accuracy = pad_list(val_accuracy, max_length)
    val_BACC = pad_list(val_BACC, max_length)
    val_AUC = pad_list(val_AUC, max_length)
    test_predictions = pad_list(test_predictions, max_length)
    test_targets = pad_list(test_targets, max_length)
    test_accuracy = pad_list(test_accuracy, max_length)
    test_recall = pad_list(test_recall, max_length)
    test_BACC = pad_list(test_BACC, max_length)
    test_AUC = pad_list(test_AUC, max_length)

    # Create a dictionary with keys corresponding to variable names and values as lists
    data_dict = {
        'train_losses': train_losses,
        'train_accuracy': train_accuracy,
        'train_BACC': train_BACC,
        'train_AUC': train_AUC,
        'val_losses': val_losses,
        'val_accuracy': val_accuracy,
        'val_BACC': val_BACC,
        'val_AUC': val_AUC,
        'test_predictions': test_predictions,
        'test_targets': test_targets,
        'test_accuracy': test_accuracy,
        'test_recall': test_recall,
        'test_BACC': test_BACC,
        'test_AUC': test_AUC
    }
    
    # Create a DataFrame from the dictionary
    Results_DF = pd.DataFrame(data_dict)
    #return train_results_df, val_results_df, test_results_df, best_model
    return Results_DF, best_model



def Train_Tabular2(train_loader, val_loader, test_loader, num_epochs = 100, patience = 50):
    
    
    take_train_batch = iter(train_loader)
    train_iter_batch = next(take_train_batch)
    input_dim = train_iter_batch[0].mordred.shape[1]

    train_iter = iter(train_loader)
    train_iter_batch = next(train_iter)
    input_size = train_iter_batch[0].mordred.shape[1]

    hid_size1 = 150
    hid_size2 = 100
    last_layer_size = 1
    
    
    mlp_model = ANN_cls(input_size, hid_size1, hid_size2, last_layer_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(mlp_model.parameters(), lr = 0.001)

    
    early_stop_counter = 0
    best_val_loss = float('inf')
    best_model = None

    
    
    train_losses = []
    train_predictions = []
    train_targets = []
    train_accuracy, train_BACC, train_AUC = [], [], []
    
    val_losses = []
    val_accuracy, val_BACC, val_AUC = [], [], []
    
   
    for epoch in range(num_epochs):
        mlp_model.train()
        total_loss = 0.0
        
        for index, batch in enumerate(train_loader):
            #X_batch = batch.mordred.reshape(-1, 1)
            #y_batch = batch.y.float().reshape(-1, 1)
            target = batch.y.float()
            
            optimizer.zero_grad()
            outputs = mlp_model(batch.mordred)
            train_loss = criterion(outputs, target.view(-1, 1).float())
         
            train_loss.backward()
            optimizer.step()
    
            total_loss += train_loss.item()*batch.mordred.size(0)  # Multiply by batch size to get total loss
            
            train_predictions.extend((outputs).detach().numpy().flatten())
            train_targets.extend(target.view(-1).cpu().numpy())
    
        # Calculate average loss for the epoch
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Calculate accuracy for training data
        train_accuracy.append(accuracy_score(train_targets, np.round(train_predictions)))
        #print(train_accuracy)
        train_BACC.append(balanced_accuracy_score(train_targets, np.round(train_predictions)))
        train_AUC.append(roc_auc_score(train_targets, train_predictions))
    
        
        mlp_model.eval()
        
        total_loss = 0.0

        with torch.no_grad():
            val_predictions = []
            val_targets = []
            for index, batch in enumerate(val_loader):
                
                #X_batch = batch.mordred.reshape(-1, 1)
                #y_batch = batch.y.float().reshape(-1, 1)
                target = batch.y.float()
                outputs = mlp_model(batch.mordred)
                val_loss = criterion(outputs, target.view(-1, 1).float())
             
                total_loss += val_loss.item()*batch.mordred.size(0)  # Multiply by batch size to get total loss
                            
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(target.view(-1).cpu().numpy())
           
            avg_val_loss = total_loss/len(val_loader.dataset)
            val_losses.append(avg_val_loss)
            val_predictions = np.concatenate(val_predictions)
            val_targets = np.concatenate(val_targets)
            epoch_val_acc = accuracy_score(val_targets, np.round(val_predictions))
            epoch_val_BACC = balanced_accuracy_score(val_targets, np.round(val_predictions))
            epoch_val_AUC = roc_auc_score(val_targets, val_predictions)
            val_accuracy.append(epoch_val_acc)
            val_BACC.append(epoch_val_BACC)
            val_AUC.append(epoch_val_AUC)
        
        
        # Update best model if the current validation RMSE is lower
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = mlp_model.state_dict()
            early_stop_counter = 0  # Reset the counter when a better model is found
        else:
            early_stop_counter += 1
    
        # Print and log to TensorBoard every 10 epochs
        if epoch % 10 == 0:

            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Balanced Accuracy: {epoch_val_BACC:.4f}, Validation AUC: {epoch_val_AUC:.4f}")
    
        # Early stopping check
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} as there is no improvement for {patience} consecutive epochs.")
            break
    
    
    
    
    # Evaluate on test data
        
    mlp_model.eval()
    test_losses = []
    test_predictions = []
    test_targets = []
    test_accuracy, test_BACC, test_AUC = [], [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            
            target = batch.y.float()
            outputs = mlp_model(batch.mordred)

            test_loss = criterion(outputs, target.view(-1, 1).float())
         
            total_loss += test_loss.item()*batch.mordred.size(0)  # Multiply by batch size to get total loss
                        
            test_predictions.append(outputs.cpu().numpy())
            test_targets.append(target.view(-1).cpu().numpy())
            
        

       
        avg_test_loss = total_loss/len(test_loader.dataset)
        test_predictions = np.concatenate(test_predictions)
        test_targets = np.concatenate(test_targets)

        test_accuracy = [accuracy_score(test_targets, np.round(test_predictions))]
        test_BACC = [balanced_accuracy_score(test_targets, np.round(test_predictions))]
        test_AUC = [roc_auc_score(test_targets, test_predictions)]
        test_recall = [recall_score(test_targets, np.round(test_predictions))]
        test_predictions = test_predictions.tolist()
        test_targets = test_targets.tolist()
        
        print(f"MLP Classifier Accuracy: {test_accuracy}")
        print(f"MLP Classifier Recall: {test_recall}")
        print(f"MLP Classifier Balanced Accuracy: {test_BACC}")
        print(f"MLP Classifier AUC: {test_AUC}")


    # Pad the shorter lists with None to make them equal in length
    max_length = max(len(train_losses), len(train_accuracy), len(train_BACC), len(train_AUC),
                  len(val_losses), len(val_accuracy), len(val_BACC), len(val_AUC),
                  len(test_predictions), len(test_targets), len(test_accuracy),
                  len(test_recall), len(test_BACC), len(test_AUC))

    def pad_list(lst, max_length, pad_value=None):
        return lst + [pad_value] * (max_length - len(lst))
    
    train_losses = pad_list(train_losses, max_length)
    train_accuracy = pad_list(train_accuracy, max_length)
    train_BACC = pad_list(train_BACC, max_length)
    train_AUC = pad_list(train_AUC, max_length)
    val_losses = pad_list(val_losses, max_length)
    val_accuracy = pad_list(val_accuracy, max_length)
    val_BACC = pad_list(val_BACC, max_length)
    val_AUC = pad_list(val_AUC, max_length)
    test_predictions = pad_list(test_predictions, max_length)
    test_targets = pad_list(test_targets, max_length)
    test_accuracy = pad_list(test_accuracy, max_length)
    test_recall = pad_list(test_recall, max_length)
    test_BACC = pad_list(test_BACC, max_length)
    test_AUC = pad_list(test_AUC, max_length)

    # Create a dictionary with keys corresponding to variable names and values as lists
    data_dict = {
        'train_losses': train_losses,
        'train_accuracy': train_accuracy,
        'train_BACC': train_BACC,
        'train_AUC': train_AUC,
        'val_losses': val_losses,
        'val_accuracy': val_accuracy,
        'val_BACC': val_BACC,
        'val_AUC': val_AUC,
        'test_predictions': test_predictions,
        'test_targets': test_targets,
        'test_accuracy': test_accuracy,
        'test_recall': test_recall,
        'test_BACC': test_BACC,
        'test_AUC': test_AUC
    }
    
    # Create a DataFrame from the dictionary
    Results_DF = pd.DataFrame(data_dict)
    #return train_results_df, val_results_df, test_results_df, best_model
    return Results_DF, best_model


""" Train Graph Neural Net"""

num_epochs = 100

def Train_Graph(graph_train_loader, graph_val_loader, graph_test_loader, num_epochs = 1000, patience = 50):
    
    train_loader, val_loader, test_loader = graph_train_loader, graph_val_loader, graph_test_loader
    
    take_train_batch = iter(train_loader)
    train_iter_batch = next(take_train_batch)
    input_dim = train_iter_batch[0].x.shape[1]
    print(input_dim)

    
    hidden_dim = 128
    output_dim = 1 
    
    model = MPNNModel(input_dim, hidden_dim, output_dim)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
        
    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

    #test model prediction
        
    train_iter = iter(train_loader)
    first_batch = next(train_iter)
    out = model(first_batch.x, first_batch.edge_index, first_batch.batch)

    
    early_stop_counter = 0
    best_val_loss = float('inf')
    best_model = None

    
    
    train_losses = []
    train_predictions = []
    train_targets = []
    train_accuracy, train_BACC, train_AUC = [], [], []
    
    val_losses = []
    val_accuracy, val_BACC, val_AUC = [], [], []
        
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
    
        # Training loop
        for index , batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(outputs, batch.y.view(-1, 1).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            
            train_predictions.extend((outputs).detach().numpy().flatten())
            train_targets.extend(batch.y.flatten().numpy())
        
        # Calculate average loss for the epoch
        
        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Calculate accuracy for training data
        train_accuracy.append(accuracy_score(train_targets, np.round(train_predictions)))
        #print(train_accuracy)
        train_BACC.append(balanced_accuracy_score(train_targets, np.round(train_predictions)))
        train_AUC.append(roc_auc_score(train_targets, train_predictions))
    
        
        model.eval()
        
        total_loss = 0.0
        
        val_predictions = []
        val_targets = []
        with torch.no_grad():
            
            for index, val_batch in enumerate(val_loader):
                val_batch = val_batch.to(device)
                val_outputs = model(val_batch.x, val_batch.edge_index, val_batch.batch)
                val_loss = criterion(val_outputs, val_batch.y.view(-1, 1).float())


                total_loss += val_loss.item()* val_batch.num_graphs  # Multiply by batch size to get total loss
                
                val_predictions.extend((val_outputs).detach().numpy().flatten())
                val_targets.extend(val_batch.y.flatten().numpy())
                

           
            avg_val_loss = total_loss/len(val_loader.dataset)
            val_losses.append(avg_val_loss)
            epoch_val_acc = accuracy_score(val_targets, np.round(val_predictions))
            epoch_val_BACC = balanced_accuracy_score(val_targets, np.round(val_predictions))
            epoch_val_AUC = roc_auc_score(val_targets, val_predictions)
            val_accuracy.append(epoch_val_acc)
            val_BACC.append(epoch_val_BACC)
            val_AUC.append(epoch_val_AUC)
        
        
        # Update best model if the current validation RMSE is lower
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            early_stop_counter = 0  # Reset the counter when a better model is found
        else:
            early_stop_counter += 1
    
        # Print and log to TensorBoard every 10 epochs
        if epoch % 10 == 0:

            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Balanced Accuracy: {epoch_val_BACC:.4f}, Validation AUC: {epoch_val_AUC:.4f}")
    
        # Early stopping check
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} as there is no improvement for {patience} consecutive epochs.")
            break
    
    
    
    
    # Evaluate on test data
        
    model.eval()
    test_losses = []
    test_predictions = []
    test_targets = []
    test_accuracy, test_BACC, test_AUC = [], [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            
            batch = batch.to(device)
            optimizer.zero_grad()

            outputs = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(outputs, batch.y.view(-1, 1).float())

            total_loss += loss.item()* batch.num_graphs  # Multiply by batch size to get total loss

            test_predictions.extend((outputs).detach().numpy().flatten())
            test_targets.extend(batch.y.flatten().numpy())

       
        avg_test_loss = total_loss/len(test_loader.dataset)
        #test_predictions = np.concatenate(test_predictions)
        #test_targets = np.concatenate(test_targets)

        test_accuracy = [accuracy_score(test_targets, np.round(test_predictions))]
        test_BACC = [balanced_accuracy_score(test_targets, np.round(test_predictions))]
        test_AUC = [roc_auc_score(test_targets, test_predictions)]
        test_recall = [recall_score(test_targets, np.round(test_predictions))]
        #test_predictions = test_predictions.tolist()
        #test_targets = test_targets.tolist()
        
        print(f"MLP Classifier Accuracy: {test_accuracy}")
        print(f"MLP Classifier Recall: {test_recall}")
        print(f"MLP Classifier Balanced Accuracy: {test_BACC}")
        print(f"MLP Classifier AUC: {test_AUC}")


    # Pad the shorter lists with None to make them equal in length
    max_length = max(len(train_losses), len(train_accuracy), len(train_BACC), len(train_AUC),
                  len(val_losses), len(val_accuracy), len(val_BACC), len(val_AUC),
                  len(test_predictions), len(test_targets), len(test_accuracy),
                  len(test_recall), len(test_BACC), len(test_AUC))

    def pad_list(lst, max_length, pad_value=None):
        return lst + [pad_value] * (max_length - len(lst))
    
    train_losses = pad_list(train_losses, max_length)
    train_accuracy = pad_list(train_accuracy, max_length)
    train_BACC = pad_list(train_BACC, max_length)
    train_AUC = pad_list(train_AUC, max_length)
    val_losses = pad_list(val_losses, max_length)
    val_accuracy = pad_list(val_accuracy, max_length)
    val_BACC = pad_list(val_BACC, max_length)
    val_AUC = pad_list(val_AUC, max_length)
    test_predictions = pad_list(test_predictions, max_length)
    test_targets = pad_list(test_targets, max_length)
    test_accuracy = pad_list(test_accuracy, max_length)
    test_recall = pad_list(test_recall, max_length)
    test_BACC = pad_list(test_BACC, max_length)
    test_AUC = pad_list(test_AUC, max_length)

    # Create a dictionary with keys corresponding to variable names and values as lists
    data_dict = {
        'train_losses': train_losses,
        'train_accuracy': train_accuracy,
        'train_BACC': train_BACC,
        'train_AUC': train_AUC,
        'val_losses': val_losses,
        'val_accuracy': val_accuracy,
        'val_BACC': val_BACC,
        'val_AUC': val_AUC,
        'test_predictions': test_predictions,
        'test_targets': test_targets,
        'test_accuracy': test_accuracy,
        'test_recall': test_recall,
        'test_BACC': test_BACC,
        'test_AUC': test_AUC
    }
    
    # Create a DataFrame from the dictionary
    Results_DF = pd.DataFrame(data_dict)
    #return train_results_df, val_results_df, test_results_df, best_model
    return Results_DF, best_model


class MPNN_FE(MessagePassing):
    def __init__(self, input_dim, hidden_dim):
        super(MPNN_FE, self).__init__(aggr='mean')  # 'mean' aggregation for global pooling
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, batch):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)

        # Global Pooling (stack different aggregations)
        x = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1)
        return x


class ANN_FE(nn.Module):
    def __init__(self, input_size, hid_size1, hid_size2):
        super(ANN_FE, self).__init__()
        self.fc1 = nn.Linear(input_size, hid_size1)
        self.act_hid = nn.ReLU()
        self.fc2 = nn.Linear(hid_size1, hid_size2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_hid(x)
        x = self.fc2(x)
        x = self.act_hid(x)
        return x


class Multi_Modal(nn.Module):
    def __init__(self, input_graph_dim, input_tabular_dim,hid_dim_graph, hid_size1, hid_size2, FE_concat_size):
        super(Multi_Modal, self).__init__()
        self.graph_FE = MPNN_FE(input_graph_dim, hid_dim_graph)
        self.tabular_FE = ANN_FE(input_tabular_dim, hid_size1, hid_size2)
        self.fc1 = nn.Linear(FE_concat_size, hid_size1)
        self.act_hid = nn.ReLU()
        self.fc2 = nn.Linear(hid_size1, hid_size2)
        self.fc3 = nn.Linear(hid_size2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_graph, edge_index, batch, x_tabular):
        x_1 = self.graph_FE(x_graph, edge_index, batch)
        x_2 = self.tabular_FE(x_tabular)
        #print(f'x_1 shape: {x_1.shape}')  # Print the shape of x_1
        #print(f'x_2 shape: {x_2.shape}')  # Print the shape of x_2
        x = torch.cat((x_1, x_2), 1)
        #print(f'Concatenated x shape: {x.shape}')  # Print the shape after concatenation
        x = self.fc1(x)
        #print(f'After fc1 shape: {x.shape}')  # Print the shape after fc1
        x = self.act_hid(x)
        x = self.fc2(x)
        #print(f'After fc2 shape: {x.shape}')  # Print the shape after fc2
        x = self.act_hid(x)
        x = self.fc3(x)
        #print(f'After fc3 shape: {x.shape}')  # Print the shape after fc3
        x = self.sigmoid(x)
        return x


def Train_graph_Tabular(train_loader, val_loader, test_loader, num_epochs=100, patience=50):

    input_graph_dim = train_loader.dataset[0].x.shape[1]
    input_tabular_dim = train_loader.dataset[0].mordred.shape[1]
    hid_dim_graph = 128
    hid_size1 = 150
    hid_size2 = 100
    FE_concat_size = hid_dim_graph*2 + hid_size2  # Concatenated feature size
    
    model = Multi_Modal(input_graph_dim, input_tabular_dim, hid_dim_graph, hid_size1, hid_size2, FE_concat_size)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    early_stop_counter = 0
    best_val_loss = float('inf')
    best_model = None

    train_losses = []
    train_accuracy, train_BACC, train_AUC = [], [], []

    val_losses = []
    val_accuracy, val_BACC, val_AUC = [], [], []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        train_predictions = []
        train_targets = []

        for index, batch in enumerate(train_loader):
            optimizer.zero_grad()
            x_graph, edge_index,  x_tabular = batch.x, batch.edge_index,  batch.mordred
            outputs = model(x_graph, edge_index, batch.batch, x_tabular)
            
            target = batch.y.float().view(-1, 1)  # Ensure target shape is (batch_size, 1)
            train_loss = criterion(outputs, target)
            train_loss.backward()
            optimizer.step()

            total_loss += train_loss.item() * batch.mordred.size(0)  # Multiply by batch size to get total loss

            train_predictions.extend(outputs.detach().numpy().flatten())
            train_targets.extend(target.view(-1).cpu().numpy())

        # Calculate average loss for the epoch
        avg_train_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        print(f'epoch Number_{epoch} average loss:', avg_train_loss)

        # Calculate accuracy for training data
        train_accuracy.append(accuracy_score(train_targets, np.round(train_predictions)))
        train_BACC.append(balanced_accuracy_score(train_targets, np.round(train_predictions)))
        train_AUC.append(roc_auc_score(train_targets, train_predictions))

        # Validation step (not shown here, but you can add it similarly to training)
        
        
        
        with torch.no_grad():
            total_loss = 0.0
            val_predictions = []
            val_targets = []
            
            for index, batch in enumerate(val_loader):
                
                
                x_graph, edge_index,  x_tabular = batch.x, batch.edge_index,  batch.mordred
                outputs = model(x_graph, edge_index, batch.batch, x_tabular)
                
                target = batch.y.float().view(-1, 1)  # Ensure target shape is (batch_size, 1)
                val_loss = criterion(outputs, target)
             
                total_loss += val_loss.item()*batch.mordred.size(0)  # Multiply by batch size to get total loss
                            
                val_predictions.append(outputs.cpu().numpy())
                val_targets.append(target.view(-1).cpu().numpy())
           
            avg_val_loss = total_loss/len(val_loader.dataset)
            val_losses.append(avg_val_loss)
            val_predictions = np.concatenate(val_predictions)
            val_targets = np.concatenate(val_targets)
            epoch_val_acc = accuracy_score(val_targets, np.round(val_predictions))
            epoch_val_BACC = balanced_accuracy_score(val_targets, np.round(val_predictions))
            epoch_val_AUC = roc_auc_score(val_targets, val_predictions)
            val_accuracy.append(epoch_val_acc)
            val_BACC.append(epoch_val_BACC)
            val_AUC.append(epoch_val_AUC)
        
        
        # Update best model if the current validation RMSE is lower
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict()
            early_stop_counter = 0  # Reset the counter when a better model is found
        else:
            early_stop_counter += 1
    
        # Print and log to TensorBoard every 10 epochs
        if epoch % 10 == 0:

            print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}, Validation Balanced Accuracy: {epoch_val_BACC:.4f}, Validation AUC: {epoch_val_AUC:.4f}")
    
        # Early stopping check
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1} as there is no improvement for {patience} consecutive epochs.")
            break
    # Evaluate on test data
        
    model.eval()
    test_predictions = []
    test_targets = []
    test_accuracy, test_BACC, test_AUC = [], [], []
    total_loss = 0.0
    
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            x_graph, edge_index,  x_tabular = batch.x, batch.edge_index,  batch.mordred
            outputs = model(x_graph, edge_index, batch.batch, x_tabular)
            
            target = batch.y.float().view(-1, 1)  # Ensure target shape is (batch_size, 1)

            test_loss = criterion(outputs, target)
         
            total_loss += test_loss.item()*batch.mordred.size(0)  # Multiply by batch size to get total loss
                        
            test_predictions.append(outputs.cpu().numpy())
            test_targets.append(target.view(-1).cpu().numpy())
            
        

       
        avg_test_loss = total_loss/len(test_loader.dataset)
        test_predictions = np.concatenate(test_predictions)
        test_targets = np.concatenate(test_targets)

        test_accuracy = [accuracy_score(test_targets, np.round(test_predictions))]
        test_BACC = [balanced_accuracy_score(test_targets, np.round(test_predictions))]
        test_AUC = [roc_auc_score(test_targets, test_predictions)]
        test_recall = [recall_score(test_targets, np.round(test_predictions))]
        test_predictions = test_predictions.tolist()
        test_targets = test_targets.tolist()
        
        print(f"MLP Classifier Accuracy: {test_accuracy}")
        print(f"MLP Classifier Recall: {test_recall}")
        print(f"MLP Classifier Balanced Accuracy: {test_BACC}")
        print(f"MLP Classifier AUC: {test_AUC}")


    # Pad the shorter lists with None to make them equal in length
    max_length = max(len(train_losses), len(train_accuracy), len(train_BACC), len(train_AUC),
                  len(val_losses), len(val_accuracy), len(val_BACC), len(val_AUC),
                  len(test_predictions), len(test_targets), len(test_accuracy),
                  len(test_recall), len(test_BACC), len(test_AUC))

    def pad_list(lst, max_length, pad_value=None):
        return lst + [pad_value] * (max_length - len(lst))
    
    train_losses = pad_list(train_losses, max_length)
    train_accuracy = pad_list(train_accuracy, max_length)
    train_BACC = pad_list(train_BACC, max_length)
    train_AUC = pad_list(train_AUC, max_length)
    val_losses = pad_list(val_losses, max_length)
    val_accuracy = pad_list(val_accuracy, max_length)
    val_BACC = pad_list(val_BACC, max_length)
    val_AUC = pad_list(val_AUC, max_length)
    test_predictions = pad_list(test_predictions, max_length)
    test_targets = pad_list(test_targets, max_length)
    test_accuracy = pad_list(test_accuracy, max_length)
    test_recall = pad_list(test_recall, max_length)
    test_BACC = pad_list(test_BACC, max_length)
    test_AUC = pad_list(test_AUC, max_length)

    # Create a dictionary with keys corresponding to variable names and values as lists
    data_dict = {
        'train_losses': train_losses,
        'train_accuracy': train_accuracy,
        'train_BACC': train_BACC,
        'train_AUC': train_AUC,
        'val_losses': val_losses,
        'val_accuracy': val_accuracy,
        'val_BACC': val_BACC,
        'val_AUC': val_AUC,
        'test_predictions': test_predictions,
        'test_targets': test_targets,
        'test_accuracy': test_accuracy,
        'test_recall': test_recall,
        'test_BACC': test_BACC,
        'test_AUC': test_AUC
    }
    
    # Create a DataFrame from the dictionary
    Results_DF = pd.DataFrame(data_dict)
    #return train_results_df, val_results_df, test_results_df, best_model
    return Results_DF, best_model
            





def Plot_Bar_ROC(results_df):
    
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['train_losses'], label='Train Loss')
    plt.plot(results_df['val_losses'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()
    
    # Plotting the training and validation BACC
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['train_BACC'], label='Train BACC')
    plt.plot(results_df['val_BACC'], label='Validation BACC')
    plt.xlabel('Epochs')
    plt.ylabel('Balanced Accuracy (BACC)')
    plt.title('Training and Validation Balanced Accuracy (BACC)')
    plt.legend()
    plt.show()
    
    
    results_df.columns
    
    Test_Accuracy = results_df['test_accuracy'][0]
    Test_Recall = results_df['test_recall'][0]
    Test_AUC =results_df['test_AUC'][0]
    # Bar plot
    
    # Bar colors
    colors = ['#4e79a7', '#2ca02c', '#98df8a']  # Three shades of blue
    colors=['#6495ED', '#4169E1', '#1E90FF']  # Adjust width as needed
    
    
    # Plotting the bars with borders
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Test Accuracy', 'Test Recall', 'Test AUC'], [Test_Accuracy, Test_Recall, Test_AUC],
                   color=colors, width=0.5, edgecolor='black')  # Adding edgecolor for borders
    
    # Adding exact values on the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', va='bottom', fontsize=18)
    
    # Adding labels and title
    plt.xlabel('Metrics', fontsize=18)
    plt.ylabel('Values', fontsize=18)
    plt.title('Performance Metrics', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    
    # Display the plot
    plt.tight_layout()
    plt.show()
    
    # Other parts of your code here...

    
    """ ROC curve plot"""
    
    y_true , y_pred = list(results_df['test_targets']), list(results_df['test_predictions'])
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    # Plotting the ROC curve with borders
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.00])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=18)
    plt.legend(loc='lower right', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.show()


def Labels_balance(Labels):
    """Plot class balance in the dataset"""
    count_1 = sum(1 for label in Labels if label == 1)
    count_0 = len(Labels)- count_1
    balance_ratio = count_1 / count_0
    
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Label 1', 'Label 0'], [count_1, count_0], color=['#cce1ff', '#66a3ff'])
    
    
    # Set plot properties
    plt.xlabel('Labels', fontsize=18)
    plt.ylabel('Counts', fontsize=18)
    plt.title(f'Balance Ratio Bar Plot :{balance_ratio:.3f}', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    
    # Show the plot
    plt.tight_layout()
    plt.show()



def Comparison_Plot_Bar_ROC(results_dfs):
    # Define colors for bars
    colors = ['#6495ED', '#4169E1', '#1E90FF']  # Adjust as needed
    
    # Initialize lists to store metric values
    acc_values, recall_values, auc_values = [], [], []
    labels = ['Test Accuracy', 'Test Recall', 'Test AUC']
    
    # Extract metric values from each dataframe
    for idx, results_df in enumerate(results_dfs):
        acc_values.append(results_df['test_accuracy'][0])
        recall_values.append(results_df['test_recall'][0])
        auc_values.append(results_df['test_AUC'][0])

    # Plotting the bars for Test Accuracy, Test Recall, and Test AUC
    plt.figure(figsize=(10, 6))
    for i, (label, acc, recall, auc) in enumerate(zip(labels, acc_values, recall_values, auc_values)):
        plt.bar([f'{label} (Dataset {i+1})'], [acc], color=colors[i], alpha=0.8, label='Test Accuracy')
        plt.bar([f'{label} (Dataset {i+1})'], [recall], color=colors[i], alpha=0.6, label='Test Recall')
        plt.bar([f'{label} (Dataset {i+1})'], [auc], color=colors[i], alpha=0.4, label='Test AUC')

    # Adding labels and title
    plt.xlabel('Metrics', fontsize=18)
    plt.ylabel('Datasets', fontsize=18)
    plt.title('Performance Metrics Comparison', fontsize=18)
    plt.legend(loc='best', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    # Display the plot
    plt.tight_layout()
    plt.show()


filename_cof_1 ="Cocrystal_Data_Screen.csv"
root_path_cof_1 = r"C:\Users\magha\OneDrive - The University of Western Ontario\ML for Cocrystal Prediction 2\Data_source\Graph_data\coformer1"
filename_cof_2 ="Cocrystal_Data_Screen.csv"
root_path_cof_2 = r"C:\Users\magha\OneDrive - The University of Western Ontario\ML for Cocrystal Prediction 2\Data_source\Graph_data\coformer2"
data_cof_1 = pd.read_csv(os.path.join(root_path_cof_1, filename_cof_1))
data_cof_2 = pd.read_csv(os.path.join(root_path_cof_2, filename_cof_2))


combined_graphs, Labels = preprocess_graph_data(root_path_cof_1, filename_cof_1, root_path_cof_2, filename_cof_2)
    
graph_train_loader, graph_val_loader, graph_test_loader = graph_split_batch(combined_graphs, Batch_size= 32, train_size=0.7, val_size=0.1, random_state=104)

X_df = Preprocess_Tabular_descriptors(root_path_cof_1, filename_cof_1, root_path_cof_2, filename_cof_2)

train_loader, test_loader, val_loader, selected_features = Mordred_split_selection(X_df, Labels, Batch_size = 32, selected_feat_no= 500, test_size=0.2, val_size=0.1, random_state=104)

Mixed_graphs, Labels = preprocess_combined_modalities(root_path_cof_1, filename_cof_1, root_path_cof_2, filename_cof_2, selected_features)

Mixed_graph_train_loader, Mixed_graph_val_loader, Mixed_graph_test_loader = graph_split_batch(Mixed_graphs, Batch_size= 32, train_size=0.7, val_size=0.1, random_state=104)

results_df_tabular, best_model_tabular = Train_Tabular2(Mixed_graph_train_loader, Mixed_graph_val_loader, Mixed_graph_test_loader)

#results_df_tabular, best_model_tabular = Train_Tabular(train_loader, val_loader, test_loader)

results_df_graph, best_graph_model = Train_Graph(graph_train_loader, graph_val_loader, graph_test_loader)

results_dF_multimodal, best_multimodal_model = Train_graph_Tabular(Mixed_graph_train_loader, Mixed_graph_val_loader, Mixed_graph_test_loader, num_epochs=100, patience=50)

results_df = results_dF_multimodal



Plot_Bar_ROC(results_dF_multimodal)
Plot_Bar_ROC(results_df_graph)
Plot_Bar_ROC(results_df_tabular)

Labels_balance(Labels)

Comparison_Plot_Bar_ROC([results_dF_multimodal, results_df_graph, results_df_tabular])

