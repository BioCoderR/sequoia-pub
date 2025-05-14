def sliding_window_method_parallel(df, patch_size_resized, 
                                   feat_model, model, inds_gene_of_interest, stride, 
                                   feat_model_type, feat_dim, model_type='vis', device='cpu', batch_size=16):
    """
    Memory-efficient sliding window method using batch processing and CUDA parallelism.
    """
    max_x = max(df['xcoord_tf'])
    max_y = max(df['ycoord_tf'])

    preds = {ind_gene: {} for ind_gene in inds_gene_of_interest}

    for x in tqdm(range(0, max_x, stride)):
        for y in range(0, max_y, stride):
            window = df[((df['xcoord_tf'] >= x) & (df['xcoord_tf'] < (x + 10))) &
                        ((df['ycoord_tf'] >= y) & (df['ycoord_tf'] < (y + 10)))]

            if window.shape[0] > ((10 * 10) / 2):
                # Get patches in batch
                features_batch = []
                valid_indices = []

                for ind in window.index:
                    col = df.iloc[ind]['xcoord']
                    row = df.iloc[ind]['ycoord']
                    if hasattr(slide, 'read_region'):
                        patch = slide.read_region((col, row), 0, (patch_size_resized, patch_size_resized)).convert('RGB')
                    else:
                        patch = Image.fromarray(np.asarray(slide)[col:col + patch_size_resized, row:row + patch_size_resized])
                    patch_tf = transforms_(patch).unsqueeze(0).to(device)
                    features_batch.append(patch_tf)
                    valid_indices.append(ind)

                    # Process in batches
                    if len(features_batch) == batch_size:
                        features_all = process_batch(features_batch, feat_model, feat_model_type, feat_dim, device)
                        update_predictions(features_all, model, model_type, inds_gene_of_interest, preds, valid_indices, device)
                        features_batch = []
                        valid_indices = []

                # Process remaining patches in the batch
                if features_batch:
                    features_all = process_batch(features_batch, feat_model, feat_model_type, feat_dim, device)
                    update_predictions(features_all, model, model_type, inds_gene_of_interest, preds, valid_indices, device)

    if stride < 10:
        # Average predictions for overlapping strides
        for ind_gene in inds_gene_of_interest:
            for key in preds[ind_gene].keys():
                preds[ind_gene][key] = np.mean(preds[ind_gene][key])

    return preds


def process_batch(features_batch, feat_model, feat_model_type, feat_dim, device):
    """
    Process a batch of patches with the feature extractor.
    """
    features_batch = torch.cat(features_batch)
    with torch.no_grad():
        if feat_model_type == 'resnet':
            features_all = feat_model.forward_extract(features_batch)
        else:
            features_all = feat_model(features_batch)

    # Pad if necessary
    if features_all.shape[0] < 100:
        padding = torch.cat([torch.zeros(1, feat_dim).to(device) for _ in range(100 - features_all.shape[0])])
        features_all = torch.cat([features_all, padding])

    return features_all


def update_predictions(features_all, model, model_type, inds_gene_of_interest, preds, valid_indices, device):
    """
    Update predictions with model predictions for the given batch.
    """
    with torch.no_grad():
        if model_type == 'he2rna':
            features_all = torch.unsqueeze(features_all, dim=0)
            features_all = rearrange(features_all, 'b c f -> b f c')
        model_predictions = model(features_all)

    predictions = model_predictions.detach().cpu().numpy()
    for i, ind in enumerate(valid_indices):
        for ind_gene in inds_gene_of_interest:
            if stride == 10:
                preds[ind_gene][ind] = predictions[i, ind_gene]
            else:
                preds[ind_gene][ind] = preds[ind_gene].get(ind, []) + [predictions[i, ind_gene]]


if __name__ == '__main__':
    print('Start running visualize script with memory efficiency')

    # ... (Rest of the script remains the same)

    ############################## Sliding window with parallelism
    folds = [int(i) for i in args.folds.split(',')]
    num_folds = len(folds)

    for fold in folds:
        # Load model for each fold
        fold_ckpt = checkpoint + 'model_best_' + str(fold) + '.pt'
        if (fold == 0) and ((args.model_type == 'vit') or (args.model_type == 'vis')):
            fold_ckpt = fold_ckpt.replace('_0', '')

        input_dim = 2048 if args.feat_type == 'resnet' else 1024
        if args.model_type == 'vit':
            model = ViT(num_outputs=len(gene_ids), dim=input_dim, depth=6, heads=16, mlp_dim=2048, dim_head=64)
        elif args.model_type == 'he2rna':
            model = HE2RNA(input_dim=input_dim, layers=[256, 256],
                           ks=[1, 2, 5, 10, 20, 50, 100],
                           output_dim=len(gene_ids), device=device)
        elif args.model_type == 'vis':
            model = ViS(num_outputs=len(gene_ids), input_dim=input_dim,
                        depth=6, nheads=16, dimensions_f=64, dimensions_c=64, dimensions_s=64, device=device)

        model.load_state_dict(torch.load(fold_ckpt, map_location=device))
        model = model.to(device)
        model.eval()

        # Get indices of requested genes
        inds_gene_of_interest = [gene_ids.index(gene_name) for gene_name in gene_names if gene_name in gene_ids]

        # Get visualization
        preds = sliding_window_method_parallel(df=df, patch_size_resized=patch_size_resized,
                                               feat_model=feat_model, model=model,
                                               inds_gene_of_interest=inds_gene_of_interest, stride=stride,
                                               feat_model_type=args.feat_type, feat_dim=input_dim, 
                                               model_type=args.model_type, device=device, batch_size=16)

        # Aggregate predictions
        for ind_gene in inds_gene_of_interest:
            res_df[gene_ids[ind_gene] + '_' + str(fold)] = res_df.index.map(preds[ind_gene])

    # Save results
    save_name = save_path + 'stride-' + str(stride) + '.csv'
    res_df.to_csv(save_name)

    print('Done with memory-efficient inference')