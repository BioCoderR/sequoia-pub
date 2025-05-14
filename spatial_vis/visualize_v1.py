import os
import argparse
from tqdm import tqdm
import json
import numpy as np
import pandas as pd
from einops import rearrange

from scipy.ndimage.morphology import binary_dilation
import openslide
from PIL import Image
import timm
import torch
from torchvision import transforms

# Added necessary imports
import pickle
import gc

# Assuming these custom modules are in the same directory or accessible via Python path
from src.he2rna import HE2RNA
from src.vit import ViT
from src.resnet import resnet50
from src.tformer_lin import ViS

BACKGROUND_THRESHOLD = .5


def read_pickle(path):
    objects = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break
    return objects


# Optimized sliding_window_method
def sliding_window_method(df, patch_size_resized, feat_model, model, inds_gene_of_interest, stride, feat_model_type, feat_dim, model_type='vis', device='cpu'):
    max_x = max(df['xcoord_tf'])
    max_y = max(df['ycoord_tf'])

    preds = {ind_gene: {} for ind_gene in inds_gene_of_interest}

    for x in tqdm(range(0, max_x, stride)):
        for y in range(0, max_y, stride):
            window = df[
                ((df['xcoord_tf'] >= x) & (df['xcoord_tf'] < (x + 10))) &
                ((df['ycoord_tf'] >= y) & (df['ycoord_tf'] < (y + 10)))
            ]

            if window.shape[0] > ((10 * 10) / 2):
                # Batch processing for patches
                patches = []
                for ind in window.index:
                    col = df.iloc[ind]['xcoord']
                    row = df.iloc[ind]['ycoord']
                    patch = slide.read_region((col, row), 0, (patch_size_resized, patch_size_resized)).convert('RGB')
                    patches.append(transforms_(patch).unsqueeze(0).to(device))

                # Process patches in batches
                batch_size = 32
                features_all = []
                for i in range(0, len(patches), batch_size):
                    batch_patches = torch.cat(patches[i:i + batch_size], dim=0)
                    with torch.no_grad():
                        if feat_model_type == 'resnet':
                            features = feat_model.forward_extract(batch_patches)
                        else:
                            features = feat_model(batch_patches)
                        features_all.append(features)

                features_all = torch.cat(features_all)
                if features_all.shape[0] < 100:
                    padding = torch.zeros(100 - features_all.shape[0], feat_dim, device=device)
                    features_all = torch.cat([features_all, padding])

                # Get predictions
                with torch.no_grad():
                    if model_type == 'he2rna':
                        features_all = rearrange(torch.unsqueeze(features_all, dim=0), 'b c f -> b f c')
                    model_predictions = model(features_all)

                predictions = model_predictions.detach().cpu().numpy()[0]

                # Aggregate predictions
                for ind_gene in inds_gene_of_interest:
                    for _, key in enumerate(window.index):
                        preds[ind_gene][key] = predictions[ind_gene]

    return preds

if __name__=='__main__':

    print('Start running visualize script')

    parser = argparse.ArgumentParser(description='Getting features')
    parser.add_argument('--study', type=str, help='cancer study abbreviation, lowercase')
    parser.add_argument('--project', type=str, help='name of project (spatial_GBM_pred, TCGA-GBM, PESO, Breast-ST)')
    parser.add_argument('--gene_names', type=str, help='name of genes to visualize, separated by commas. if you want all the predicted genes, pass "all" ')
    parser.add_argument('--wsi_file_name', type=str, help='wsi filename')
    parser.add_argument('--save_folder', type=str, help='destination folder')
    parser.add_argument('--slide_path', type=str, default='/projects/conco/gundla/root/uniglacier/models/trained/image2st/Patches_hdf5/',help='Path to the directory containing slide HDF5 files')
    parser.add_argument('--mask_path', type=str,default='/projects/conco/gundla/root/uniglacier/models/trained/image2st/Patches_hdf5/',help='Path to the directory containing mask files')
    parser.add_argument('--model_type', type=str, help='model to use: "he2rna", "vit" or "vis"')
    parser.add_argument('--feat_type', type=str, help='"resnet" or "uni"')
    parser.add_argument('--folds', type=str, help='folds to use in prediction split by comma', default='0,1,2,3,4')
    args = parser.parse_args()

    study = args.study
    assert args.feat_type in ['resnet', 'uni']
    assert args.model_type in ['vit', 'vis', 'he2rna']

    checkpoint = f'{args.model_type}_{args.feat_type}/{study}/'
    # Assuming test_results.pkl contains a list of objects, and we need the first one.
    obj = read_pickle(checkpoint + 'test_results.pkl')[0]
    gene_ids = obj['genes']

    stride = 1
    patch_size = 256
    wsi_file_name = args.wsi_file_name
    project = args.project
    slide_id = os.path.splitext(os.path.basename(wsi_file_name))[0]
    save_path = os.path.join(args.save_folder, slide_id)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.gene_names != 'all':
        if '.npy' in args.gene_names:
            gene_names = np.load(args.gene_names,allow_pickle=True)
        else:
            gene_names = args.gene_names.split(",")
    else:
        gene_names = gene_ids

    if 'TCGA' in wsi_file_name:
        slide_path = args.slide_path
        mask_path = args.mask_path
        wsi_basename = os.path.basename(wsi_file_name).replace('.svs', '')
        mask_file_path = os.path.join(mask_path, wsi_basename, 'mask.npy')
        manual_resize = None
    elif project == 'spatial_GBM_pred':
        slide_path_root = f'./Spatial_GBM/pyramid/'
        mask_path_root = './Spatial_GBM/masks/'
        px_df = pd.read_csv('./Spatial_Heiland/data/classify/spot_diameter.csv')
        mask = np.load(mask_path_root + wsi_file_name.replace('.tif', '.npy'))
        diam = px_df[px_df['slide_id']==wsi_file_name.split('_')[1] + '_T']['pixel_diameter'].values[0]
        um_px = 55/diam
        manual_resize = 0.5/um_px
    elif project == 'Breast-ST':
        slide_path_root = './Gen-Pred/Breast-ST/wsis/'
        mask_path_root = './Gen-Pred/Breast-ST/masks/'
        mask = np.load(mask_path_root+wsi_file_name.replace('.tif', '.npy'))
        metadata = json.load(open(f"./Gen-Pred/Breast-ST/metadata/{wsi_file_name.replace('.tif','.json')}"))
        mag = eval(metadata['magnification'].replace('x',''))
        manual_resize = mag/20.0
    else:
        print('please provide correct file name format (containing "TCGA") or correct project id ("spatial_GBM_pred" or "Breast-ST")')
        exit()

    slide = openslide.OpenSlide(slide_path_root + wsi_file_name)
    downsample_factor = int(slide.dimensions[0]/mask.shape[1]) # mask.shape[0] was width, mask.shape[1] was height based on transpose later
                                                              # slide.dimensions[0] is width. If mask is (height, width) then mask.shape[1] is width.
                                                              # original: downsample_factor = int(slide.dimensions[0]/mask.shape[0])
                                                              # if mask is (H,W) after load, and slide.dimensions is (W,H)
                                                              # then W_slide / W_mask = slide.dimensions[0] / mask.shape[1]
                                                              # or H_slide / H_mask = slide.dimensions[1] / mask.shape[0]
                                                              # Given mask = (np.transpose(mask, axes=[1,0]))*1, original mask is (width, height) as loaded by np.load
                                                              # So mask.shape[0] is width. This seems correct.
                                                              # Let's assume mask is loaded (H,W) then transposed to (W,H)
                                                              # If mask is loaded as (H,W), mask.shape[0] is H, mask.shape[1] is W.
                                                              # slide.dimensions[0] is W_slide, slide.dimensions[1] is H_slide.
                                                              # If mask is transposed to (W,H), then mask.shape[0] is W_mask, mask.shape[1] is H_mask.
                                                              # So original: W_slide / W_mask. This seems fine.

    slide_dim0, slide_dim1 = slide.dimensions[0], slide.dimensions[1]

    if manual_resize is None:
        resize_factor = float(slide.properties.get('aperio.AppMag',20)) / 20.0
    else:
        resize_factor = manual_resize

    patch_size_resized = int(resize_factor * patch_size)
    patch_size_in_mask = int(patch_size_resized/downsample_factor)
    if patch_size_in_mask == 0: patch_size_in_mask = 1 # Ensure patch_size_in_mask is at least 1

    valid_idx = []
    # Transpose mask to be (width, height) for easier indexing with (col, row)
    # Original code had mask = (np.transpose(mask, axes=[1,0]))*1
    # This means the loop expects mask[col_idx, row_idx] effectively if mask was originally (H,W)
    # Let's assume mask is loaded as (rows, cols) i.e. (height, width) by np.load
    # Then mask[row, col] is standard. The original code's transpose suggests it wanted mask[col,row] iteration.
    # If mask is (H,W), then mask[row_downs:..., col_downs:...] is natural.
    # The loop "for col ... for row ..." implies col is x-like and row is y-like.
    # slide.read_region((col, row)...) col=x, row=y.
    # So, if mask is (H,W), we'd index mask[row_downs, col_downs]. The transpose was likely to align this.
    # Let's keep the transpose logic for now as it was in the original.
    mask_transposed = (np.transpose(mask, axes=[1,0]))*1 # mask_transposed is (width, height)

    for col in tqdm(range(0, slide_dim0 - patch_size_resized, patch_size_resized), desc="Finding Valid Patches (Cols)"):
        for row in range(0, slide_dim1 - patch_size_resized, patch_size_resized):
            row_downs = int(row / downsample_factor)
            col_downs = int(col / downsample_factor)
            
            # Ensure indices are within bounds for the transposed mask
            if row_downs + patch_size_in_mask > mask_transposed.shape[1] or \
               col_downs + patch_size_in_mask > mask_transposed.shape[0]:
                continue

            # Original used mask[row_downs:row_downs+patch_size_in_mask,col_downs:col_downs+patch_size_in_mask]
            # After transpose, this would be mask_transposed[col_downs:col_downs+patch_size_in_mask, row_downs:row_downs+patch_size_in_mask]
            patch_in_mask = mask_transposed[col_downs : col_downs + patch_size_in_mask,
                                            row_downs : row_downs + patch_size_in_mask]
            
            # binary_dilation can be slow on very large masks if iterations is high.
            # Consider applying dilation to the whole mask once if possible, though patch-wise might be necessary.
            patch_in_mask = binary_dilation(patch_in_mask, iterations=3)

            if patch_in_mask.sum() >= (BACKGROUND_THRESHOLD * patch_in_mask.size):
                valid_idx.append((col, row))
    
    del mask # Release memory for the raw mask if large
    del mask_transposed
    gc.collect()

    if not valid_idx:
        print("No valid patches found. Exiting.")
        exit()

    df = pd.DataFrame(valid_idx, columns=['xcoord', 'ycoord'])
    # Potential dtype optimization for DataFrames (example, adjust as needed):
    # df['xcoord'] = df['xcoord'].astype(np.uint32) # Or int32 if values can be negative (not typical for coords)
    # df['ycoord'] = df['ycoord'].astype(np.uint32)

    df['xcoord_tf'] = ((df['xcoord'] - min(df['xcoord'])) / patch_size_resized).astype(np.int32) # Use int32, or int16 if range allows
    df['ycoord_tf'] = ((df['ycoord'] - min(df['ycoord'])) / patch_size_resized).astype(np.int32) # Use int32, or int16 if range allows
    print('Got dataframe of valid tiles')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.feat_type == 'resnet':
        transforms_ = transforms.Compose([
            transforms.Resize((256,256)), # Original was (256,265) - typo? Assuming 256x256
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        feat_model = resnet50(pretrained=True).to(device)
        feat_model.eval()
    else: # 'uni'
        feat_model = timm.create_model("vit_large_patch16_224", img_size=224,
                                       patch_size=16, init_values=1e-5,
                                       num_classes=0, dynamic_img_size=True)
        local_dir = "/projects/conco/gundla/root/uniglacier/models/pretrained/UNI"
        feat_model.load_state_dict(torch.load(os.path.join(local_dir,
                                    "pytorch_model.bin"), map_location=device), strict=True)
        transforms_ = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        feat_model = feat_model.to(device)
        feat_model.eval()

    res_df = df.copy(deep=True)
    folds = [int(i) for i in args.folds.split(',')]
    checkpoint_base = '/projects/conco/gundla/root/uniglacier/models/pretrained/sequoia/'
    for fold_num, fold in enumerate(folds):
        print(f"Processing fold {fold + 1}/{len(folds)}")
        fold_ckpt_path = os.path.join(checkpoint_base, f'sequoia-gbm-{fold}', 'model.safetensors')
        if (fold == 0) and ((args.model_type == 'vit') or (args.model_type == 'vis')):
            fold_ckpt_path = fold_ckpt_path.replace('_0','')

        input_dim = 2048 if args.feat_type == 'resnet' else 1024
        model_pred = None # Initialize model
        if args.model_type == 'vit':
            model_pred = ViT(num_outputs=len(gene_ids),
                             dim=input_dim, depth=6, heads=16, mlp_dim=2048, dim_head = 64)
            model_pred.load_state_dict(torch.load(fold_ckpt_path, map_location=torch.device(device)))
        elif args.model_type == 'he2rna':
            model_pred = HE2RNA(input_dim=input_dim, layers=[256,256],
                                ks=[1,2,5,10,20,50,100],
                                output_dim=len(gene_ids), device=device) # Pass device here
            fold_ckpt_path = fold_ckpt_path.replace('best_','') # model specific naming
            # Assuming HE2RNA.load_state_dict can take state_dict directly or model object's state_dict
            loaded_state = torch.load(fold_ckpt_path, map_location=torch.device(device))
            if hasattr(loaded_state, 'state_dict'):
                 model_pred.load_state_dict(loaded_state.state_dict())
            else: # If it's already a state_dict
                 model_pred.load_state_dict(loaded_state)

        elif args.model_type == 'vis':
            model_pred = ViS(num_outputs=len(gene_ids),
                             input_dim=input_dim,
                             depth=6, nheads=16,
                             dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device) # Pass device here
            from safetensors.torch import load_file
            state_dict = load_file(fold_ckpt_path)
            model_pred.load_state_dict(state_dict, map_location=torch.device(device))

        model_pred = model_pred.to(device)
        model_pred.eval()

        inds_gene_of_interest = []
        current_gene_names_to_process = []
        for gene_name in gene_names:
            try:
                inds_gene_of_interest.append(gene_ids.index(gene_name))
                current_gene_names_to_process.append(gene_name)
            except ValueError:
                print(f'Warning: Gene {gene_name} not in predicted gene_ids. Skipping.')
        
        if not inds_gene_of_interest:
            print("No valid genes of interest found for this fold. Skipping prediction.")
            continue

        # Pass slide and transforms_
        preds = sliding_window_method(df=df, patch_size_resized=patch_size_resized,
                                      feat_model=feat_model, model=model_pred,
                                      inds_gene_of_interest=inds_gene_of_interest, stride=stride,
                                      feat_model_type=args.feat_type, feat_dim=input_dim,
                                      slide=slide, transforms_=transforms_, # Pass them here
                                      model_type=args.model_type, device=device)

        for i, ind_gene in enumerate(inds_gene_of_interest):
            gene_name_original = current_gene_names_to_process[i] # Use the actual name for column
            # Map predictions; preds[ind_gene] is a dict {tile_key: prediction_value}
            # res_df.index should align with the keys used in preds if df was not re-indexed.
            # The keys in preds[ind_gene] are indices from the original `df`.
            res_df[gene_name_original + '_' + str(fold)] = res_df.index.map(preds[ind_gene])
            # Fill NaNs that might result from map if some tile_keys were not in preds[ind_gene] (e.g. not enough tissue in window)
            # This might happen if a tile was in `df` but no window covering it passed the threshold.
            # Or, if a gene had no predictions in `preds` (handled by `final_preds` init).
            # A fill_value for .map or a .fillna() might be needed depending on desired behavior for missing preds.
            # For now, let's assume map works or NaNs are acceptable if a tile truly gets no prediction.


        del preds # Explicitly delete large intermediate dict
        del model_pred # Release model from memory for this fold
        if device == torch.device('cuda'):
            torch.cuda.empty_cache() # Clear CUDA cache
        gc.collect() # Trigger garbage collection

    # Calculate mean across folds
    for gene_idx, gene_name_in_ids in enumerate(gene_ids): # Iterate through all possible genes
        if gene_name_in_ids in gene_names: # Only process genes that were requested
            fold_cols = [gene_name_in_ids + '_' + str(f) for f in folds if gene_name_in_ids + '_' + str(f) in res_df.columns]
            if fold_cols: # Ensure there are columns to average
                 res_df[gene_name_in_ids] = res_df[fold_cols].mean(axis=1)
            # else: # Gene was requested but no fold produced results for it (e.g. due to earlier skips)
            #    res_df[gene_name_in_ids] = np.nan # Or some other placeholder

    save_name = save_path + 'stride-' + str(stride) + '.parquet'
    try:
        res_df.to_parquet(save_name, engine='pyarrow', compression='snappy') # Or compression='gzip' for better compression
        print(f'Results saved to Parquet: {save_name} (using snappy compression)')
    except ImportError:
        print("PyArrow not installed. Falling back to Gzipped CSV. Consider 'pip install pyarrow' for Parquet support.")
        print(f'Results saved to {save_name}')
        save_name_csv_gz = save_path + 'stride-' + str(stride) + '.csv.gz'
        res_df.to_csv(save_name_csv_gz, index=False, compression='gzip') # index=False is common for Parquet, can be true for CSV
        print(f'Results saved to Gzipped CSV: {save_name_csv_gz}')
    
    print('Done')
