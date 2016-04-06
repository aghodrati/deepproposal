% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------
function objectness_model = train_objectness_i(scale_id, layer_id, win_sizes, net_gpu, opts)
% scale_id: image scale for training
% layer_id: Feature layer to be used
%
%   Trains an objectness for DeepProposal.

cache_dir = [opts.model.objectness '/l' num2str(layer_id) '/'];
if ~exist(cache_dir, 'dir') mkdir(cache_dir); end;

%%% load ground-truth bboxes
load(opts.imdb.trn_info_path, 'gtids','recs');

%%% print parameters
fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Training options:\n');
disp(opts.train);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Training Objecness for scale:%d and layer:%d\n', opts.scales(scale_id), layer_id);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');

%%%%%%%%%%%%%%%%%%%%%
%training
%%%%%%%%%%%%%%%%%%%%%
% Init training caches
init_cache_path = sprintf([cache_dir 'cache_s%d_0.mat'], scale_id);
if ~exist(init_cache_path, 'file')
    [X_pos, keys_pos, X_neg, keys_neg] = sample_features_init(net_gpu, recs, gtids, win_sizes, scale_id, layer_id, opts);
    caches = init_cache(X_pos, keys_pos, X_neg, keys_neg);
    %save(init_cache_path, 'caches'); 
else
    fprintf('-prc: loading cache . . .');
    load(init_cache_path, 'caches'); 
end

% Train with hard negative mining
first_time = true;
first_epoch = true;
max_hard_epochs = 1; %% one pass over the data is enough

for hard_epoch = 1:max_hard_epochs
  for i = 0 %:length(ids) %0 for initial model
    fprintf('-prc: hard neg epoch: %d/%d image: %d/%d cache_siz_neg:%d \n', hard_epoch, max_hard_epochs, i, length(gtids), size(caches.X_neg,1));

    % Get positives and negatives of image bboxes
    if ~first_time
        [X_neg, keys] = sample_features(objectness_model, caches, i, net_gpu, recs, gtids, win_sizes, scale_id, layer_id, opts);
        %make sure that duplicates has been removed
        if (~isempty(keys))
            if ~isempty(caches.keys_neg)
                [~, ~, dups] = intersect(caches.keys_neg, keys, 'rows');
                assert(isempty(dups));
            end
        end    
    
        % Add sampled negatives to training cache
        caches.X_neg = cat(1, caches.X_neg, X_neg);
        caches.keys_neg = cat(1, caches.keys_neg, keys);
        caches.num_added = caches.num_added + size(keys,1);
    end

    % Update model if
    %  - first time seeing negatives
    %  - more than retrain_limit negatives have been added
    %  - its the final image of the final epoch
    is_last_time = (hard_epoch == max_hard_epochs && i == length(gtids));
    hit_retrain_limit = (caches.num_added > caches.retrain_limit);
    if (first_time || hit_retrain_limit || is_last_time) && ~isempty(caches.X_neg)
        fprintf('Bfore Pruning easy negatives: Cache holds %d pos examples %d neg examples\n', ...
                size(caches.X_pos,1), size(caches.X_neg,1));
        

        neg_ixs = []; %1:40000;
        pos_ixs = []; %1:15000;

        [new_w, new_b] = update_model(caches, opts, pos_ixs, neg_ixs);
        objectness_model.detectors.W = new_w;
        objectness_model.detectors.B = new_b;
        caches.num_added = 0;

        z_pos = caches.X_pos * new_w + new_b;
        z_neg = caches.X_neg * new_w + new_b;
        caches.pos_loss(end+1) = opts.train.svm_C * opts.train.pos_loss_weight * sum(max(0, 1 - z_pos));
        caches.neg_loss(end+1) = opts.train.svm_C * sum(max(0, 1 + z_neg));
        caches.reg_loss(end+1) = 0.5 * new_w' * new_w + 0.5 * (new_b / (opts.train.bias_mult+eps))^2;
        caches.tot_loss(end+1) = caches.pos_loss(end) + caches.neg_loss(end) + caches.reg_loss(end);

        t = length(caches.tot_loss);
        fprintf('    %2d: obj val: %.3f = %.3f (pos) + %.3f (neg) + %.3f (reg)\n', ...
                  t, caches.tot_loss(t), caches.pos_loss(t), ...
                  caches.neg_loss(t), caches.reg_loss(t));

        % store negative support vectors for visualizing later
        SVs_neg = find(z_neg > -1 - eps);
        objectness_model.SVs.keys_neg = caches.keys_neg(SVs_neg, :);
        objectness_model.SVs.scores_neg = z_neg(SVs_neg);

        % evict easy examples
        easy = find(z_neg < caches.evict_thresh);
        caches.X_neg(easy,:) = [];
        caches.keys_neg(easy,:) = [];
        fprintf('After Pruning easy negatives:  Cache holds %d pos examples %d neg examples\n', ...
                size(caches.X_pos,1), size(caches.X_neg,1));
        fprintf('  %d pos support vectors, ', numel(find(z_pos <  1 + eps)));
        fprintf('  %d neg support vectors\n', numel(find(z_neg > -1 - eps)));
    end    
    
    first_time = false;
    
  end %id
  first_epoch = false;
  
end %epoch

% save the final objectness_model
save(opts.train.model_name, 'objectness_model');
% ------------------------------------------------------------------------



% ------------------------------------------------------------------------
function [w, b] = update_model(cache, opts, pos_inds, neg_inds)
% ------------------------------------------------------------------------
solver = 'liblinear';
liblinear_type = 2; % l2 regularized l1 hinge loss

if ~exist('pos_inds', 'var') || isempty(pos_inds)
  num_pos = size(cache.X_pos, 1);
  pos_inds = 1:num_pos;
else
  num_pos = length(pos_inds);
  fprintf('[subset mode] using %d out of %d total positives\n', ...
      num_pos, size(cache.X_pos,1));
end
if ~exist('neg_inds', 'var') || isempty(neg_inds)
  num_neg = size(cache.X_neg, 1);
  neg_inds = 1:num_neg;
else
  num_neg = length(neg_inds);
  fprintf('[subset mode] using %d out of %d total negatives\n', ...
      num_neg, size(cache.X_neg,1));
end

switch solver
  case 'liblinear'
    %pos_loss_weight = num_neg./(num_pos+num_neg);
    %ll_opts = sprintf('-w1 %.5f -w0 %.5f -c %.5f -s %d -B 10 -q', pos_loss_weight, 1-pos_loss_weight, opts.train.svm_C, liblinear_type);
    ll_opts = sprintf('-w1 %.5f -w0 %.5f -c %.5f -s %d -B %.5f -q', opts.train.pos_loss_weight, 1, opts.train.svm_C, liblinear_type, opts.train.bias_mult);    
    X = sparse(size(cache.X_pos,2), num_pos+num_neg);
    X(:,1:num_pos) = cache.X_pos(pos_inds,:)';
    X(:,num_pos+1:end) = cache.X_neg(neg_inds,:)';
    y = cat(1, ones(num_pos,1), zeros(num_neg,1));
    llm = train(y, X, ll_opts, 'col');
    w = single(llm.w(1:end-1)');
    b = single(llm.w(end)*opts.train.bias_mult);
  
  otherwise
    error('unknown solver: %s', solver);
end


% ------------------------------------------------------------------------
function cache = init_cache(X_pos, keys_pos, X_neg, key_neg)
% ------------------------------------------------------------------------
cache.X_pos = X_pos;
cache.X_neg = X_neg;
cache.keys_neg = key_neg;
cache.keys_pos = keys_pos;
cache.num_added = size(X_neg,1);
cache.retrain_limit = 50000;
cache.evict_thresh = -1.2;
cache.hard_thresh = -1.0001;
cache.pos_loss = [];
cache.neg_loss = [];
cache.reg_loss = [];
cache.tot_loss = [];


% ------------------------------------------------------------------------
function [X_neg, keys] = sample_features(mdl, caches, ind, net, gt_recs, ids, siz_win, scale_id, lid, opts)
% ------------------------------------------------------------------------
max_nneg = 1000; %opts.train.max_nneg;
neg_ov_thre = opts.train.neg_ov_thre;

im = imread(sprintf('%s/%s.%s', opts.imdb.imgpath_trn, ids{ind}, opts.imdb.img_ext));    
x_feat_map = compute_featmaps(im, net, opts.scales(scale_id), lid);
    
%fname = sprintf(opts.train.feat_path, ids{ind});
%load(fname,'res_sc'); 
%x_feat_map = res_sc{scale_id};

siz_box = [size(x_feat_map,2) size(x_feat_map,1)];
im_siz = gt_recs(ind).imgsize;
[props_all, props] = gen_props2(im_siz(1:2), siz_box, siz_win); %[c1 r1 c1+w r1+h]

feats_intg = integral_feats2(x_feat_map, props);

%find positive and negative indices
ov_cover = false(size(props_all,1), 1);
for j=1:length(gt_recs(ind).objects)
    if gt_recs(ind).objects(j).difficult == 1, continue; end;
    bbgt = gt_recs(ind).objects(j).bbox;
    ov1 = find_overlap_all(props_all , bbgt);
    %[~, idx] = sort(ov1,'descend');
    ov_nocover_obj = ov1<=neg_ov_thre;
    ov_cover = ov_cover | ~ov_nocover_obj;
    
end
neg_ix = find(~ov_cover);
nneg = min(length(neg_ix), max_nneg);
neg_ix_rand = randperm(length(neg_ix), nneg);
neg_ix_rand = neg_ix(neg_ix_rand);

X_neg = feats_intg(neg_ix_rand,:);

% Find hard negatives
z = bsxfun(@plus, X_neg*mdl.detectors.W, mdl.detectors.B);
I = find(z > caches.hard_thresh);    
key_vals = neg_ix_rand(I);

% Avoid adding duplicate features
keys_ = [ind*ones(length(key_vals),1) key_vals];
if ~isempty(caches.keys_neg) && ~isempty(keys_)
  [~, ~, dups] = intersect(caches.keys_neg, keys_, 'rows');
  keep = setdiff(1:size(keys_,1), dups);
  I = I(keep);
end

% Unique hard negatives
X_neg = X_neg(I,:);
key_vals = neg_ix_rand(I);
keys = [ind*ones(length(key_vals),1) key_vals];


% ------------------------------------------------------------------------
function [X_pos, keys_pos, X_neg, keys_neg] = sample_features_init(net, gt_recs, ids, siz_win, scale_id, lid, opts)
% ------------------------------------------------------------------------
max_npos = opts.train.max_npos;
max_nneg = opts.train.max_nneg;
pos_ov_thre = opts.train.pos_ov_thre;
neg_ov_thre = opts.train.neg_ov_thre;

nimgs = length(ids);
d = opts.train.d;
%n_obj_apprx = nimgs*5;
n_obj=0; for i=1:nimgs, for j=1:length(gt_recs(i).objects), n_obj=n_obj+1; end; end;
X_pos = zeros(max_npos * n_obj, d, 'single');
X_neg = zeros(max_nneg * nimgs, d, 'single');
keys_pos = zeros(max_npos * n_obj, 2, 'single');
keys_neg = zeros(max_nneg * nimgs, 2, 'single');
pix=1; nix=1;

rng(0);
ix_rand = randperm(nimgs, min(opts.train.nimg_rand, nimgs));

tic
for i=1:length(ix_rand) 
    ind = ix_rand(i);
    
    im = imread(sprintf('%s/%s.%s', opts.imdb.imgpath_trn, ids{ind}, opts.imdb.img_ext));    
    x_feat_map = compute_featmaps(im, net, opts.scales(scale_id), lid);
    
    %fname = sprintf(opts.feat_path, ids{ind});
    %load(fname,'res_sc');    
    %x_feat_map = res_sc{scale_id};
    
    siz_box = [size(x_feat_map,2) size(x_feat_map,1)];
    im_siz = gt_recs(ind).imgsize;
    
    [props_all, props] = gen_props2(im_siz(1:2), siz_box, siz_win); %[x y x+w y+h]
    
    %find positive and negative indices
    pos_ix=[]; ov_cover=false(size(props_all,1), 1);
    nobj=1;
    for j=1:length(gt_recs(ind).objects)
        if gt_recs(ind).objects(j).difficult == 1, continue; end;
        bbgt = double(gt_recs(ind).objects(j).bbox);
                
        ov1 = find_overlap_all(props_all , bbgt);
        [~, idx] = sort(ov1,'descend');
        ov_cover_obj = ov1>=pos_ov_thre;
        ov_nocover_obj = ov1<neg_ov_thre; % & ov1>0.1;
        ov_cover = ov_cover | ~ov_nocover_obj;
        npos = min(sum(ov_cover_obj), max_npos);
        pos_ix = cat( 1, pos_ix, idx(1:npos) );
        
        nobj = nobj+1;
    end
    neg_ix = find(~ov_cover);
    nneg = min(length(neg_ix), max_nneg);
    neg_ix_rand = randperm(length(neg_ix), nneg);
    neg_ix_rand = neg_ix(neg_ix_rand);

    feats_intg = integral_feats2(x_feat_map, props([neg_ix_rand; pos_ix],:));
    %feats_intg = integral_feats_sp(x_feat_map, props([neg_ix_rand; pos_ix],:), 2); %TODO: in case of spatial pyramid pooling

    nneg = length(neg_ix_rand);
    X_neg_i = feats_intg(1:nneg,:);
    X_pos_i = feats_intg(nneg+1:end,:);

    npos=size(X_pos_i,1);
    X_pos(pix:pix+npos-1,:) = X_pos_i;
    keys_pos(pix:pix+npos-1,:) = [ind*ones(npos,1) pos_ix];
    pix=pix+npos;
        
    nneg=length(neg_ix_rand);
    X_neg(nix:nix+nneg-1,:) = X_neg_i;    
    keys_neg(nix:nix+nneg-1,:) = [ind*ones(nneg,1) neg_ix_rand];    
    nix=nix+nneg;
    
    % display progress
    if toc>10
        fprintf('pr: collecting features: %d/%d\n',i, length(ix_rand));
        drawnow; tic;
    end
end
X_pos(pix:end,:) = [];
X_neg(nix:end,:) = [];
keys_pos(pix:end,:) = [];
keys_neg(nix:end,:) = [];


% ------------------------------------------------------------------------
function x_map = compute_featmaps(im, net, scale, lid)
% ------------------------------------------------------------------------
averageImage = single(net.meta.normalization.averageImage);
im_avg = mean(mean(averageImage,1),2);

%get image statistics
[h, w, d] =  size(im);
if d==1, im=cat(3, im, im, im); end;
im_siz = [h,w];      
[~,smin] = min(im_siz);
im_s = single(im);

%extract feature maps
ss = scale ./ im_siz(smin);
siz_scale = round(im_siz.*ss);
im_ = imresize(im_s, siz_scale);
im_ = im_ - repmat(im_avg, siz_scale);
%gpu
im_gpu_ = gpuArray(im_);
res = vl_simplenn(net, im_gpu_);
%res = vl_simplenn(net, im_); %cpu

x_map = gather(res(lid+1).x); %the first one is input

clear res im_gpu_
