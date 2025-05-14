import os
import argparse
from tqdm import tqdm
import json
import numpy as np
import cupy as cp
import cudf
from einops import rearrange
from scipy.ndimage import binary_dilation
import openslide
from PIL import Image
import timm
import torch
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor

from src.he2rna import HE2RNA
from src.vit import ViT
from src.resnet import resnet50
from src.tformer_lin import ViS

BACKGROUND_THRESHOLD = 0.5


def read_pickle(path):
    with open(path, "rb") as openfile:
        return pickle.load(openfile)


def extract_features(feat_model, patch_tf, feat_model_type, device):
    with torch.no_grad():
        if feat_model_type == 'resnet':
            features = feat_model.forward_extract(patch_tf)
        else:
            features = feat_model(patch_tf)
    return features.cpu().numpy()


def sliding_window_method(df, patch_size_resized, feat_model, model,
                          inds_gene_of_interest, stride, feat_model_type,
                          feat_dim, model_type='vis', device='cuda'):

    max_x, max_y = int(df['xcoord_tf'].max()), int(df['ycoord_tf'].max())

    preds = {ind_gene: {} for ind_gene in inds_gene_of_interest}

    for x in tqdm(range(0, max_x, stride)):
        for y in range(0, max_y, stride):

            window = df.query(f'xcoord_tf >= {x} and xcoord_tf < {x+10} and '
                              f'ycoord_tf >= {y} and ycoord_tf < {y+10}')

            if len(window) > 50:  # optimized check
                features_all = []

                for _, row in window.iterrows():
                    col, row_coord = int(row['xcoord']), int(row['ycoord'])
                    patch = slide.read_region((col, row_coord), 0,
                                              (patch_size_resized, patch_size_resized)).convert('RGB')

                    patch_tf = transforms_(patch).unsqueeze(0).to(device)
                    features_all.append(extract_features(feat_model, patch_tf, feat_model_type, device))

                features_all = cp.vstack(features_all)

                if features_all.shape[0] < 100:
                    padding = cp.zeros((100 - features_all.shape[0], feat_dim))
                    features_all = cp.vstack([features_all, padding])

                with torch.no_grad():
                    features_tensor = torch.tensor(features_all, device=device, dtype=torch.float32)
                    if model_type == 'he2rna':
                        features_tensor = rearrange(features_tensor.unsqueeze(0), 'b c f -> b f c')
                    model_predictions = model(features_tensor)

                predictions = model_predictions.cpu().numpy()[0]

                for ind_gene in inds_gene_of_interest:
                    pred_value = predictions[ind_gene]
                    for key in window.index:
                        if stride == 10:
                            preds[ind_gene][key] = pred_value
                        else:
                            preds[ind_gene].setdefault(key, []).append(pred_value)

    if stride < 10:
        for ind_gene, values in preds.items():
            for key, val_list in values.items():
                preds[ind_gene][key] = np.mean(val_list)

    return preds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimized visualization')
    parser.add_argument('--study', type=str)
    parser.add_argument('--project', type=str)
    parser.add_argument('--gene_names', type=str)
    parser.add_argument('--wsi_file_name', type=str)
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--model_type', type=str, default='vis')
    parser.add_argument('--feat_type', type=str, default= 'uni')
    parser.add_argument('--slide_path', type=str, default='/projects/conco/gundla/root/uniglacier/models/trained/image2st/Patches_hdf5/',help='Path to the directory containing slide HDF5 files')
    parser.add_argument('--mask_path', type=str,default='/projects/conco/gundla/root/uniglacier/models/trained/image2st/Patches_hdf5/',help='Path to the directory containing mask files')
    parser.add_argument('--folds', type=str, default='0,1,2,3,4')
    args = parser.parse_args()

    device = torch.device('cuda')
    checkpoint = f'{args.model_type}_{args.feat_type}/{study}/'
    obj = read_pickle(checkpoint + 'test_results.pkl')[0]
    gene_ids = obj['genes']
    
    stride = 1
    patch_size = 256

    save_path = f'{args.save_folder}/{args.wsi_file_name}/'
    os.makedirs(save_path, exist_ok=True)
    
    if args.gene_names != 'all':
        if '.npy' in args.gene_names:
            gene_names = np.load(args.gene_names,allow_pickle=True)
        else:
            gene_names = args.gene_names.split(",")
    else:
        gene_names = gene_ids
    
    slide = openslide.OpenSlide(args.slide_path)
    mask = cp.array(np.load(args.mask_path))
    downsample_factor = slide.dimensions[0] // mask.shape[0]
    resize_factor = float(slide.properties.get('aperio.AppMag', 20)) / 20.0
    patch_size_resized = int(resize_factor * patch_size)
    patch_size_in_mask = patch_size_resized // downsample_factor

    valid_idx = [(col, row) for col in range(0, slide.dimensions[0] - patch_size_resized, patch_size_resized)
                 for row in range(0, slide.dimensions[1] - patch_size_resized, patch_size_resized)
                 if binary_dilation(mask[row//downsample_factor:(row+patch_size_resized)//downsample_factor,
                                         col//downsample_factor:(col+patch_size_resized)//downsample_factor], iterations=3).sum() >= BACKGROUND_THRESHOLD * patch_size_in_mask**2]

    df = cudf.DataFrame(valid_idx, columns=['xcoord', 'ycoord'])
    df['xcoord_tf'] = ((df['xcoord'] - df['xcoord'].min()) / patch_size_resized).astype(int)
    df['ycoord_tf'] = ((df['ycoord'] - df['ycoord'].min()) / patch_size_resized).astype(int)

    feat_model = timm.create_model("vit_large_patch16_224", img_size=224, patch_size=16, num_classes=0).to(device)
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

    results = []
    input_dim = 1024 if args.feat_type == 'uni' else 2048
    with ThreadPoolExecutor() as executor:
        futures = []            
        for fold in map(int, args.folds.split(',')):
            checkpoint_base = '/projects/conco/gundla/root/uniglacier/models/pretrained/sequoia/'
            fold_ckpt = os.path.join(checkpoint_base, f'sequoia-gbm-{fold}', 'model.safetensors')
            model = ViS(num_outputs=len(gene_ids),
                             input_dim=input_dim,
                             depth=6, nheads=16,
                             dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device) # Pass device here
            from safetensors.torch import load_file
            state_dict = load_file(fold_ckpt)
            model.load_state_dict(state_dict, map_location=torch.device(device))
            model = model.to(device)
            model.eval()
            futures.append(executor.submit(sliding_window_method, df, patch_size_resized,
                                           feat_model, model, list(range(len(gene_ids))),
                                           stride, args.feat_type, 1024, args.model_type, device))

        for future in tqdm(futures):
            results.append(future.result())

    res_df = cudf.concat([df] + [cudf.DataFrame(r) for r in results], axis=1)
    res_df.to_csv(f'{save_path}stride-{stride}.csv', index=False)

    print('Done')
