import os
import json
import argparse
import pickle
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.read_data import SuperTileRNADataset
from src.utils import filter_no_features, custom_collate_fn
from src.vit import train, evaluate, predict
from src.tformer_lin import ViS
from safetensors.torch import load_file as safe_load

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Getting features')
    parser.add_argument('--ref_file', type=str, required=True, help='Reference file')
    parser.add_argument('--feature_path', type=str, default='', help='Directory where pre-processed WSI features are stored')
    parser.add_argument('--feature_use', type=str, default='cluster_mean_features', help='Which feature to use for training the model')
    parser.add_argument('--folds', type=int, default=5, help='Folds for pre-trained model')
    parser.add_argument('--seed', type=int, default=99, help='Seed for random generation')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--depth', type=int, default=6, help='Transformer depth')
    parser.add_argument('--num-heads', type=int, default=16, help='Number of attention heads')
    parser.add_argument('--tcga_project', type=str, default='', help='The tcga_project we want to use')
    parser.add_argument('--save_dir', type=str, default='', help='Where to save results')
    parser.add_argument('--exp_name', type=str, default='exp', help='Experiment name')

    ############################################## variables ##############################################
    
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    print(device)

    ############################################## saving ##############################################

    save_dir = os.path.join(args.save_dir, args.exp_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ############################################## data prep ##############################################

    path_csv = args.ref_file
    df = pd.read_csv(path_csv)

    # filter out WSIs for which we don't have features and filter on TCGA project
    df = filter_no_features(df, feature_path = args.feature_path, feature_name = args.feature_use)
    print("After feature filtering:", len(df))
    genes = [c[4:] for c in df.columns if "rna_" in c]
    if 'tcga_project' in df.columns and args.tcga_project:
        df = df[df['tcga_project'].isin([args.tcga_project])].reset_index(drop=True)
        print(f"After TCGA project filtering ({args.tcga_project}):", len(df))
    # init test dataloader
    test_dataset = SuperTileRNADataset(df, args.feature_path, args.feature_use)
    test_dataloader = DataLoader(test_dataset, 
                                num_workers=0, pin_memory=True, 
                                shuffle=False, batch_size=args.batch_size,
                                collate_fn=custom_collate_fn)
    feature_dim = test_dataset.feature_dim

    res_preds   = []
    res_random  = []
    # cancer      = args.tcga_project.split('-')[-1].lower()

    for fold in range(args.folds):
        
        model_path = f'/projects/conco/gundla/root/uniglacier/models/pretrained/sequoia/sequoia-gbm-{fold}'
        model = ViS(
            num_outputs=467, 
            input_dim=1024, 
            depth=args.depth, 
            nheads=args.num_heads,  
            dimensions_f=64, 
            dimensions_c=64, 
            dimensions_s=64, 
            device=device)
        
        from safetensors.torch import load_file as safe_load
        state_dict = safe_load(os.path.join(model_path,"model.safetensors"))
        filtered_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and model.state_dict()[k].shape == v.shape}
        model.load_state_dict(filtered_dict, strict=False) # no_strict due to less no of genes
        model.to(device)

        # model prediction on test set
        preds, wsis, projs = predict(model, test_dataloader, run=None)

        # random predictions
        random_model = ViS(num_outputs=test_dataset.num_genes, 
                            input_dim=feature_dim, 
                            depth=args.depth, nheads=args.num_heads,  
                            dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device)
        random_model.to(device)
        random_preds, _, _ = predict(random_model, test_dataloader, run=None)

        # save predictions
        res_preds.append(preds)
        res_random.append(random_preds)

    # calculate average across folds
    avg_preds = np.mean(res_preds, axis = 0)
    avg_random = np.mean(res_random, axis = 0)

    df_pred = pd.DataFrame(avg_preds, index = wsis, columns = genes)
    df_random = pd.DataFrame(avg_random, index = wsis, columns = genes)

    test_results = {'pred': df_pred, 'random': df_random}

    with open(os.path.join(save_dir, 'test_results.pkl'), 'wb') as f:
        pickle.dump(test_results, f, protocol=pickle.HIGHEST_PROTOCOL)