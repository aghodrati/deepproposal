% check setup instructions in readme

%%%%%%%%%%%%%%%%%%%%%
%set parameters and load models
%%%%%%%%%%%%%%%%%%%%%
%add necessary paths
addpath('./deps/edges-master/');
addpath(genpath('./deps/piotr_toolbox_V3.40/toolbox/'));
run('./deps/matconvnet-1.0-beta8/matlab/vl_setupnn');

%models
opts.model.contour = './models/contour/modelF_C2.mat';
opts.model.cnn = './models/cnn/imagenet-caffe-ref.mat';
opts.model.objectness = './models/objectness/';

%set parameters
opts.thr_s = 0.5; 
opts.thr_e = opts.thr_s; %if using adaptive nms introduced in "What makes for effective detection proposals?", arXiv 2015.
opts.nbox_s1 = 4000; %%6000;
opts.nbox_s2 = 3000; %4000;
opts.nbox_s3 = 1000;
opts.layers = [14 10 6];
opts.scales = [227 300 400 600];
opts.nsliding_win = 50;
opts.step_siz = (opts.thr_s/opts.thr_e).^(1/opts.nbox_s3);

%load windows sizes obtained based on algorithm explained in sec3.1 of the paper
X = load('./best_siz_win_AR_VOC07.mat','siz_win');
win_sizes_i = X.siz_win(1:opts.nsliding_win);
for i=1:length(opts.scales), win_sizes{i} = win_sizes_i; end;

%cnn
net = load(opts.model.cnn);
net_gpu = vl_simplenn_move(net, 'gpu');

%load objectness models
mdl_obj = train_objectness(win_sizes, net_gpu, opts);

%load contour model
mdl_contour = load(opts.model.contour); 
mdl_contour = mdl_contour.model;


%%%%%%%%%%%%%%%%%%%%%
%main process
%%%%%%%%%%%%%%%%%%%%%
%read image
im1 = imread('cameraman.tif');

%compute feature maps
x_map = compute_featmaps(im1, net_gpu, opts);

%entry to the deepProposal
gen_time = tic;
boxes = deepProposal( im1, x_map, mdl_obj, mdl_contour, win_sizes, opts );
ptime = toc(gen_time);
nbox = size(boxes, 1);
fprintf('-processing one image in %0.4f sec\n', ptime);  

%%%%%%%%%%%%%%%%%%%%%
%visualization
%%%%%%%%%%%%%%%%%%%%%
%visualization of proposals    
n_box_show = 5;
fprintf('-showing the first %d boxes\n', n_box_show);
%show first n_box_show boxes in the image
subplot(1,2,1), imshow(im1);
xp=boxes(1:min(n_box_show,nbox), 1:4); xp(:,[3 4])= xp(:,[3 4])-xp(:,[1 2])+1;
colors=[];
for r=1:size(xp,1), colors=cat(1,colors, rand(1,3)); rectangle('Position', xp(r,:), 'EdgeColor', colors(r,:), 'LineWidth', 3); end;
%heatmap of boxes
im_heat = zeros(size(im1,1), size(im1,2)); im1=double(im1);
for o=1:n_box_show %size(boxes_i,1)
    bb=boxes(o,:);
    im_heat(bb(2):bb(4), bb(1):bb(3)) = im_heat(bb(2):bb(4), bb(1):bb(3)) + im1(bb(2):bb(4), bb(1):bb(3));
end
subplot(1,2,2), imshow(im_heat,[]);
