function opts = get_opts( dname )

opts.imdb.dname = dname;
if strcmp(opts.imdb.dname, 'VOC07')
    opts.imdb.dataset_path = '/esat/fluorite/aghodrat/det/VOCdevkit/'; %TODO
elseif strcmp(opts.imdb.dname, 'COCO14')
    opts.imdb.dataset_path = '/esat/fluorite/aghodrat/coco/'; %TODO
    opts.imdb.dataset_apipath = '/users/visics/aghodrat/codes/coco/atlabAPI/'; %TODO
end

%add necessary paths
addpath('./deps/edges-master/');
addpath(genpath('./deps/piotr_toolbox_V3.40/toolbox/'));
run('./deps/matconvnet/matlab/vl_setupnn');

%models
opts.model.contour = './models/contour/modelF_C2.mat';
opts.model.cnn = './models/cnn/imagenet-caffe-ref.mat';
opts.model.objectness = './models/objectness/';

%set paths
opts.feat_path = sprintf('./feats/%s/', dname); %can be a symbolic links to another place;
opts.logs_path = './logs/';
opts.outs_path  = sprintf('./outputs/%s/', dname);
if ~exist(opts.feat_path, 'dir'), mkdir(opts.feat_path); end;
if ~exist(opts.logs_path, 'dir'), mkdir(opts.logs_path); end;
if ~exist(opts.outs_path, 'dir'), mkdir(opts.outs_path); end;

%set parameters
opts.thr_s = 0.7; 
opts.thr_e = opts.thr_s; %if using adaptive nms introduced in "What makes for effective detection proposals?", arXiv 2015.
opts.nbox_s1 = 4000; %6000;
opts.nbox_s2 = 3000; %4000;
opts.nbox_s3 = 1000;
opts.layers = [14 10 6];
opts.scales = [227 300 400 600];
opts.nsliding_win = 50;
opts.step_siz = (opts.thr_s/opts.thr_e).^(1/opts.nbox_s3);

%arrange dataset
opts = arrange_imdb(opts);

fprintf('\n\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n');
fprintf('Proposal options:\n');
disp(opts);
fprintf('~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n');


end

