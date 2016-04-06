% script for extracting object proposals using DeepProposal for VOC07 and COCO14.
% check readme for setup instructions

close all;
clear

dname = 'VOC07'; %'COCO14'; %'VOC07'; 
extract_featmaps = 1; %you can either extract features on the fly (extract_featmaps=1) or load them from disk (extract_featmaps=0)
do_visualization = 0;

opts = get_opts(dname);
ninter_show = 50;

%record a log
timestamp = datestr(datevec(now()), 'dd.mmm.yyyy:HH.MM.SS');
diary_file = [opts.logs_path 'proposals_' dname '_' timestamp '.txt'];
diary(diary_file);
fprintf('Logging output in %s\n', diary_file);


%load name and ground-truth locations of test data (bboxes for evaluation)
load(opts.imdb.tst_info_path, 'gtids','recs');

%load windows sizes obtained based on algorithm explained in sec3.1 of the paper
X = load('./best_siz_win_AR_VOC07.mat','siz_win');
win_sizes_i = X.siz_win(1:opts.nsliding_win);
for i=1:length(opts.scales), win_sizes{i} = win_sizes_i; end;

%measure recall@n
recall_at = [1 3 10 30 50 100 200 300 700 1000];
%save processing time of algorithm
gtime=0; res_inf = cell(length(recall_at),1); np_m_k = zeros(length(recall_at),1); n=0;
boxes = cell(length(gtids),1); %keep proposals of all the images


%%%%%%%%%%%%%%%%%%%%%
%load models
%%%%%%%%%%%%%%%%%%%%%
%cnn
net = load(opts.model.cnn);
net_gpu = vl_simplenn_move(net, 'gpu');

%load/train objectness models
mdl_obj = train_objectness(win_sizes, net_gpu, opts);

%load contour model
mdl_contour = load(opts.model.contour); 
mdl_contour = mdl_contour.model;


%%%%%%%%%%%%%%%%%%%%%
%main loop - iterate over images
%%%%%%%%%%%%%%%%%%%%%
for ind=1:length(gtids)    
    %get image
    im1 = imread(sprintf('%s/%s.%s', opts.imdb.imgpath_tst, gtids{ind}, opts.imdb.img_ext));
    n = n + 1;
    
    %load/compute feature map for the image
    if extract_featmaps == 0
        load([opts.feat_path gtids{ind} '.mat'], 'x_map');
    else
        x_map = compute_featmaps(im1, net_gpu, opts);
        %save([opts.feat_path gtids{ind} '.mat'], 'x_map'); %TODO
    end
    
    %entry to the deepProposal
    gen_time = tic;
    boxes_i = deepProposal( im1, x_map, mdl_obj, mdl_contour, win_sizes, opts );
    gtime = gtime + toc(gen_time);
    boxes{ind,1} = boxes_i;
    nbox = size(boxes_i, 1);
    
    %visualization of proposals    
    if do_visualization==1    
        n_box_show = 20;
        %show first 10 boxes in the image
        subplot(1,2,1), imshow(im1);
        xp=boxes_i(1:min(n_box_show,nbox), 1:4); xp(:,[3 4])= xp(:,[3 4])-xp(:,[1 2])+1;
        colors=[];
        for r=1:size(xp,1), colors=cat(1,colors, rand(1,3)); rectangle('Position', xp(r,:), 'EdgeColor', colors(r,:), 'LineWidth', 3); end;
        %heatmap of boxes
        im_heat = zeros(size(im1,1), size(im1,2));
        for o=1:n_box_show %size(boxes_i,1)
            bb=boxes_i(o,:);
            im_heat(bb(2):bb(4), bb(1):bb(3)) = im_heat(bb(2):bb(4), bb(1):bb(3)) + im1(bb(2):bb(4), bb(1):bb(3));
        end
        subplot(1,2,2), imshow(im_heat,[]);
        
        stop_here=1;
        pause();
    end
    
    %%%%%%%%%%%%%%%%%%%%%
    %evaluation
    %%%%%%%%%%%%%%%%%%%%%    
    %show results ater processing every n images
     if rem(n,ninter_show)==0
        fprintf('---------------pr: %d/%d (%0.4f sec/image)-------------------------\n', n, length(gtids), gtime/n);        
     end
    recall_at_k = zeros(length(recall_at), 11); %in 11 overlap threshold
    AR_at_k = zeros(length(recall_at),1);
    %measure metrics in different number of proposals    
    for k=1:length(recall_at)
        nmax = min(recall_at(k), nbox);
        props_org_all = boxes_i(1:nmax,1:4);
        np = size(props_org_all,1);
        np_m_k(k) = np_m_k(k) + np;
        
        %compute overlap with ground-truth
        ov_i=[]; nj=1;
        for j=1:length(recs(ind).objects) %iterate over all objects in image
            bbgt = double(recs(ind).objects(j).bbox); %[x y x+w y+h]
            ov = find_overlap_all(props_org_all , bbgt);
            ov_i(nj,2) = double(recs(ind).objects(j).difficult); %is hard
            ov_i(nj,6) = max(ov); %maximum overlap
            nj=nj+1;
        end
        res_inf{k} = cat(1, res_inf{k}, ov_i);

        %display progress
        ov_all = res_inf{k};
        if rem(n, ninter_show)==0
            res_k = res_inf{k};
            xx=res_k(:,6); xx(res_k(:,2)>0) = [];
            %average recall (Hosang et al.[15]).
            [overlap_thr, recall_at_k(k,:), AR_at_k(k)] = compute_average_recall(xx);        
            nprop_avg(k) = round(np_m_k(k)/ind);
        end        
    end %over different number of proposals
    
    if rem(n,ninter_show)==0
        fprintf('overlap_thr= %.1f  %.1f  %.1f  %.1f  %.1f \n', overlap_thr(6:10))
        for k=1:length(recall_at)
            fprintf('recall@%04d= %03d  %03d  %03d  %03d  %03d  (#p:%05d)\n', recall_at(k), round(recall_at_k(k,6:10)*100), nprop_avg(k))  
        end
        fprintf('\n');
        %fprintf('AR@k= %03d', round(AR_at_k*100))
        fprintf('\n');
    end    
end

%save in format of [x y w h]
bbs=boxes; for ind=1:length(boxes), bbs{ind}(:,3:4)=bbs{ind}(:,3:4)-bbs{ind}(:,1:2)+1; end;
save(sprintf('%s/DeepProposal-a%.2f-%.2f-m%d-test.mat', opts.outs_path, opts.thr_s, opts.thr_e, opts.nbox_s3), 'bbs');

