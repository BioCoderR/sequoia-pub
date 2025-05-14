import os
import psutil
import pickle
import argparse
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from einops import rearrange
import sys
from pathlib import Path
from scipy.ndimage import binary_dilation  # Updated import
import openslide
from PIL import Image
import timm
import torch
from torchvision import transforms
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.he2rna import HE2RNA
from src.vit import ViT
from src.resnet import resnet50
from src.tformer_lin import ViS

BACKGROUND_THRESHOLD = .5

def read_pickle(path):
    """Helper function to read pickle files"""
    try:
        with open(path, "rb") as openfile:
            objects = []
            while True:
                try:
                    objects.append(pickle.load(openfile))
                except EOFError:
                    break
        return objects
    except Exception as e:
        print(f"Error reading pickle file at {path}: {e}")
        raise

def sliding_window_method(df, patch_size_resized, 
                            feat_model, model, inds_gene_of_interest, stride, 
                            feat_model_type, feat_dim, model_type='vis', device='cpu', tile_batch=100):

    max_x = max(df['xcoord_tf'])
    max_y = max(df['ycoord_tf'])

    preds = {} # {key:value} where key is a gene index and value is a new dict that contains the predictions per tile for that gene
    for ind_gene in inds_gene_of_interest:
        preds[ind_gene] = {}

    for x in tqdm(range(0, max_x, stride)):
        for y in range(0, max_y, stride):
            window = df[((df['xcoord_tf']>=x) & (df['xcoord_tf']<(x+10))) &
                        ((df['ycoord_tf']>=y) & (df['ycoord_tf']<(y+10)))]
            if window.shape[0] > ((10*10)/2):
                features_all = []
                patch_indices = []
                for ind in window.index:
                    col = df.iloc[ind]['xcoord']
                    row = df.iloc[ind]['ycoord']
                    try:
                        if hasattr(slide, 'read_region'):
                            patch = slide.read_region((col, row), 0, (patch_size_resized, patch_size_resized)).convert('RGB')
                        else:
                            patch = Image.fromarray(np.asarray(slide)[row:row+patch_size_resized, col:col+patch_size_resized])
                    except Exception as e:
                        print(f"Skipping patch at ({col}, {row}) due to read error: {e}")
                        continue
                    patch_tf = transforms_(patch).unsqueeze(0).to(device)
                    with torch.no_grad():
                        if feat_model_type == 'resnet':
                            features = feat_model.forward_extract(patch_tf)
                        else:
                            features = feat_model(patch_tf)
                    features_all.append(features)
                    patch_indices.append(ind)
                if len(features_all) == 0:
                    continue
                features_all = torch.cat(features_all)
                # Pad to 100 if needed (legacy behavior)
                if features_all.shape[0] < 100:
                    padding = torch.cat([torch.zeros(1, feat_dim) for _ in range(100-features_all.shape[0])]).to(device)
                    features_all = torch.cat([features_all, padding])

                # Mini-batch prediction
                preds_batch = []
                with torch.no_grad():
                    for start in range(0, features_all.shape[0], tile_batch):
                        mini_batch = features_all[start:start+tile_batch]
                        if model_type == 'he2rna':
                            mini_batch = mini_batch.unsqueeze(0)
                            mini_batch = rearrange(mini_batch, 'b c f -> b f c')
                        mini_preds = model(mini_batch)
                        preds_batch.append(mini_preds.detach().cpu())
                model_predictions = torch.cat(preds_batch, dim=0)

                # Only use the number of predictions corresponding to the actual patches (not padding)
                predictions = model_predictions[:len(patch_indices)]
                predictions = predictions.cpu().numpy()

                for i, key in enumerate(window.index):
                    for idx_gene, ind_gene in enumerate(inds_gene_of_interest):
                        if stride == 10:
                            preds[ind_gene][key] = predictions[i][ind_gene]
                        else:
                            if key not in preds[ind_gene]:
                                preds[ind_gene][key] = [predictions[i][ind_gene]]
                            else:
                                preds[ind_gene][key].append(predictions[i][ind_gene])

    if stride < 10:
        for ind_gene in inds_gene_of_interest:
            for key in preds[ind_gene].keys():
                preds[ind_gene][key] = np.mean(preds[ind_gene][key])

    return preds

if __name__=='__main__':

    print('Start running visualize script')

    ############################## get args
    parser = argparse.ArgumentParser(description='Getting features')
    parser.add_argument('--study', type=str, help='cancer study abbreviation, lowercase')
    parser.add_argument('--project', type=str, help='name of project (spatial_GBM_pred, TCGA-GBM, PESO, Breast-ST)')
    parser.add_argument('--gene_names', type=str, help='name of genes to visualize, separated by commas. if you want all the predicted genes, pass "all" ')
    parser.add_argument('--wsi_file_name', type=str, help='wsi filename')
    parser.add_argument('--save_folder', type=str, help='destination folder')
    parser.add_argument('--model_type', type=str, help='model to use:  "he2rna", "vit" or "vis"')
    parser.add_argument('--feat_type', type=str, help='"resnet" or "uni"')
    parser.add_argument('--folds', type=str, help='folds to use in prediction split by comma', default='0,1,2,3,4')
    parser.add_argument('--slide_path', type=str, default='/projects/conco/gundla/root/uniglacier/models/trained/image2st/Patches_hdf5/',help='Path to the directory containing slide HDF5 files')
    parser.add_argument('--mask_path', type=str,default='/projects/conco/gundla/root/uniglacier/models/trained/image2st/Patches_hdf5/',help='Path to the directory containing mask files')
    parser.add_argument('--checkpoint', type=str, help='path to checkpoint')
    # parser.add_argument('--tile_batch', type=int, default=100, help='Max patches per batch')
    args = parser.parse_args()

    ############################## general 
    study = args.study
    assert args.feat_type in ['resnet', 'uni']
    assert args.model_type in ['vit', 'vis', 'he2rna']

    ############################## get model and genes
    checkpoint = args.checkpoint
    if not checkpoint.endswith('/'):
        checkpoint += '/'

    # Load gene list from evaluation/gene_list.csv
    gene_list_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'evaluation', 'gene_list.csv')
    print(f"Loading gene list from: {gene_list_path}")

    try:
        gene_df = pd.read_csv(gene_list_path)
        gene_ids = gene_df['gene'].tolist()  # Assuming the column is named 'gene'
        print(f"Loaded {len(gene_ids)} genes from gene_list.csv")
        print("First few genes:", gene_ids[:5])
    except Exception as e:
        print(f"Error loading gene list: {e}")
        raise

    if args.gene_names != 'all':
        if '.npy' in args.gene_names:
            gene_names = np.load(args.gene_names, allow_pickle=True)
        else:
            gene_names = args.gene_names.split(",")
    else:
        gene_names = gene_ids

    ############################## prepare data
    stride = 1 #chan
    patch_size = 256 # at 20x (0.5um pp)
    wsi_file_name = args.wsi_file_name
    project = args.project 
    # Refactor output directory logic to use only the filename (without duplicating directories)
    slide_id = os.path.splitext(os.path.basename(wsi_file_name))[0]
    save_path = os.path.join(args.save_folder, slide_id)
    os.makedirs(save_path, exist_ok=True)
        
    # prepare and load WSI
    if 'TCGA' in wsi_file_name:
        slide_path = args.slide_path
        mask_path = args.mask_path
        wsi_basename = os.path.basename(wsi_file_name).replace('.svs', '')
        mask_file_path = os.path.join(mask_path, wsi_basename, 'mask.npy')
        print(f"Attempting to load mask from: {mask_file_path}")
        if not os.path.exists(mask_file_path):
            print(f"[SKIP] Mask file not found for {wsi_file_name}, skipping slide.")
            if not os.path.exists('processing_summary.csv'):
                pd.DataFrame({'processed': pd.Series(dtype=str), 'skipped': pd.Series([wsi_file_name])}).to_csv('processing_summary.csv', index=False)
            else:
                df_summary = pd.read_csv('processing_summary.csv')
                df_summary = pd.concat([
                    df_summary,
                    pd.DataFrame({'processed': pd.Series(dtype=str), 'skipped': pd.Series([wsi_file_name])})
                ])
                df_summary.to_csv('processing_summary.csv', index=False)
            sys.exit(0)
        try:
            mask = np.load(mask_file_path)
        except Exception as e:
            print(f"Error loading mask: {e}")
            raise
        manual_resize = None # nr of um/px can be read from slide properties
    elif project == 'spatial_GBM_pred':
        slide_path = f'./Spatial_GBM/pyramid/' 
        mask_path = './Spatial_GBM/masks/'
        px_df = pd.read_csv('./Spatial_Heiland/data/classify/spot_diameter.csv')
        mask = np.load(mask_path + wsi_file_name.replace('.tif', '.npy'))
        diam = px_df[px_df['slide_id']==wsi_file_name.split('_')[1] + '_T']['pixel_diameter'].values[0]
        um_px = 55/diam # um/px for the WSI
        manual_resize = 0.5/um_px
    elif project == 'Breast-ST':
        slide_path = './Gen-Pred/Breast-ST/wsis/'
        mask_path = './Gen-Pred/Breast-ST/masks/'
        mask = np.load(mask_path+wsi_file_name.replace('.tif', '.npy'))
        metadata = json.load(open(f"./Gen-Pred/Breast-ST/metadata/{wsi_file_name.replace('.tif','.json')}"))
        mag = eval(metadata['magnification'].replace('x',''))
        manual_resize = mag/20.0
    else:
        print('please provide correct file name format (containing "TCGA") or correct project id ("spatial_GBM_pred" or "Breast-ST")')
        exit()
    
    # load wsi and calculate patch size in original image (coordinates are at level 0 for openslide) and in mask
    slide = openslide.OpenSlide(os.path.join(slide_path, wsi_file_name))
    downsample_factor = int(slide.dimensions[0]/mask.shape[0]) # mask downsample factor
    slide_dim0, slide_dim1 = slide.dimensions[0], slide.dimensions[1]

    if manual_resize == None:
        resize_factor = float(slide.properties.get('aperio.AppMag',20)) / 20.0
    else:
        resize_factor = manual_resize

    patch_size_resized = int(resize_factor * patch_size)  
    patch_size_in_mask = int(patch_size_resized/downsample_factor)

    # get valid coordinates (that have tissue)
    valid_idx = []
    mask = (np.transpose(mask, axes=[1,0]))*1
    for col in range(0, slide_dim0-patch_size_resized, patch_size_resized): #slide.dimensions is (width, height)
        for row in range(0, slide_dim1-patch_size_resized, patch_size_resized):

            row_downs = int(row/downsample_factor)
            col_downs = int(col/downsample_factor)

            patch_in_mask = mask[row_downs:row_downs+patch_size_in_mask,col_downs:col_downs+patch_size_in_mask]
            patch_in_mask = binary_dilation(patch_in_mask, iterations=3)

            if patch_in_mask.sum() >= (BACKGROUND_THRESHOLD * patch_in_mask.size): 
                # keep patch
                valid_idx.append((col, row))

    # dataframe which contains coordinates of valid patches
    df = pd.DataFrame(valid_idx, columns=['xcoord', 'ycoord'])
    # rescale coordinates to (0,0) and step size 1
    df['xcoord_tf'] = ((df['xcoord']-min(df['xcoord']))/patch_size_resized).astype(int)
    df['ycoord_tf'] = ((df['ycoord']-min(df['ycoord']))/patch_size_resized).astype(int)

    print('Got dataframe of valid tiles')

    ############################## feature extractor model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.feat_type == 'resnet':
        transforms_ = transforms.Compose([
            transforms.Resize((256,265)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        feat_model = resnet50(pretrained=True).to(device)
        feat_model.eval()
    else:
        feat_model = timm.create_model("vit_large_patch16_224", img_size=224, 
                                        patch_size=16, init_values=1e-5, 
                                        num_classes=0, dynamic_img_size=True)
        local_dir = "/projects/conco/gundla/root/uniglacier/models/pretrained/UNI/"
        feat_model.load_state_dict(torch.load(os.path.join(local_dir, 
                                    "pytorch_model.bin"), map_location=device), strict=True)
        transforms_ = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        feat_model = feat_model.to(device)
        feat_model.eval()

    ############################## get preds
    res_df = df.copy(deep=True)
    folds = [int(i) for i in args.folds.split(',')]
    num_folds = len(folds)
    checkpoint_base = '/projects/conco/gundla/root/uniglacier/models/pretrained/sequoia/'

    import gc
    for fold in folds:
        fold_ckpt = os.path.join(checkpoint_base, f'sequoia-gbm-{fold}', 'model.safetensors')
        if (fold == 0) and ((args.model_type == 'vit') or (args.model_type == 'vis')):
            fold_ckpt = fold_ckpt.replace('_0','')

        input_dim = 2048 if args.feat_type == 'resnet' else 1024
        if args.model_type == 'vit':
            model = ViT(num_outputs=len(gene_ids), 
                            dim=input_dim, depth=6, heads=16, mlp_dim=2048, dim_head = 64)
            model.load_state_dict(torch.load(fold_ckpt, map_location=torch.device(device)))
        elif args.model_type == 'he2rna':
            model = HE2RNA(input_dim=input_dim, layers=[256,256],
                                ks=[1,2,5,10,20,50,100],
                                output_dim=len(gene_ids), device=device)
            fold_ckpt = fold_ckpt.replace('best_','')
            model.load_state_dict(torch.load(fold_ckpt, map_location=torch.device(device)).state_dict())
        elif args.model_type == 'vis':
            model = ViS(num_outputs=20820, 
                        input_dim=input_dim, 
                        depth=6, nheads=16,  
                        dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device)
            from safetensors.torch import load_file
            state_dict = load_file(fold_ckpt)
            model.load_state_dict(state_dict)

        model = model.to(device)
        model.eval()

        # get indices of requested genes
        inds_gene_of_interest = []
        for gene in gene_names:
            if gene in gene_ids:
                inds_gene_of_interest.append(gene_ids.index(gene))
            else:
                print(f"[WARN] Gene '{gene}' not found in gene_ids list.")

        # get visualization
        preds = sliding_window_method(
            df=df,
            patch_size_resized=patch_size_resized,
            feat_model=feat_model, model=model,
            inds_gene_of_interest=inds_gene_of_interest, stride=stride,
            feat_model_type=args.feat_type, feat_dim=input_dim, model_type=args.model_type, device=device
        )

        temp_results = {}
        for ind_gene in inds_gene_of_interest:
            temp_results[gene_ids[ind_gene] + '_' + str(fold)] = res_df.index.map(preds[ind_gene])

        res_df = pd.concat([res_df, pd.DataFrame(temp_results)], axis=1)

    for ind_gene in inds_gene_of_interest:
        res_df[gene_ids[ind_gene]] = res_df[[gene_ids[ind_gene] + '_' + str(i) for i in folds]].mean(axis=1)

    save_name = os.path.join(save_path, f'{wsi_file_name}_stride-{stride}.csv')
    print(f"[INFO] Saved predictions to: {save_name}")
    res_df.to_csv(save_name)

    print('Done')

    # Final memory cleanup
    del slide
    del df
    del res_df
    gc.collect()
    torch.cuda.empty_cache()

    if not os.path.exists('processing_summary.csv'):
        pd.DataFrame({'processed': pd.Series([wsi_file_name]), 'skipped': pd.Series(dtype=str)}).to_csv('processing_summary.csv', index=False)
    else:
        df_summary = pd.read_csv('processing_summary.csv')
        df_summary = pd.concat([
            df_summary,
            pd.DataFrame({'processed': pd.Series([wsi_file_name]), 'skipped': pd.Series(dtype=str)})
        ])
        df_summary.to_csv('processing_summary.csv', index=False)

    print('Done')