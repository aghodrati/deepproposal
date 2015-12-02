% I have included best_siz_win_AR_VOC07.mat which contains best 50 window
% sizes. This is the code for generating it.

function ov_all = win_siz_analysis()

dname = 'VOC07';
opts = get_opts(dname);

optwin.ov_thre = 0.7;
optwin.scales = opts.scales;
optwin.layers = opts.layers(1);
optwin.nimg_rand = 300;
nscale = length(optwin.scales);

load(opts.imdb.trn_info_path, 'gtids','recs');

net = load(opts.model.cnn);
net_gpu = vl_simplenn_move(net, 'gpu');

n=1; ov_all=[]; g_time=0; np_m=0; nn=0;

count=1; siz_win_all=[];
for ind=1:20
    for j=1:20
        siz_win_all{count} = [ind j];
        count = count+1;    
    end
end

tic
IoU_rng = [0.5 0.6 0.7 0.8 0.9];
siz_hist_obj_ov = zeros(40000, length(siz_win_all), length(IoU_rng));

rng(0);
ix_rand = randperm(length(gtids), min(optwin.nimg_rand,length(gtids)));

l = optwin.layers;
for i=1:optwin.nimg_rand
    ind = ix_rand(i);
    
    im1 = imread(sprintf('%s/%s.%s', opts.imdb.imgpath_trn, gtids{ind}, opts.imdb.img_ext));
    im_siz = [size(im1,1) size(im1,2)];    
    props_all = cell(nscale,1);
    
    
    x_map = compute_featmaps(im1, net_gpu, optwin);    
    gen_time = tic;
    for s=1:nscale
        siz_box = [size(x_map{s,l},2) size(x_map{s,l},1)];
        props_org = gen_props3(im_siz(2:-1:1), siz_box, siz_win_all);
        props_org(:,6) = s;
        props_all{s} = props_org;
    end    
    props_all1 = cell2mat(props_all);
    props_pos = props_all1(:,1:4);
    props_size = props_all1(:,5);
    g_time = g_time + toc(gen_time);
    
    np = size(props_pos,1);    
    np_m = np_m + np;
    nn=nn+1;
    
    %overlap with gt
    for j=1:length(recs(ind).objects)
        if recs(ind).objects(j).difficult==1, continue; end;
        
        bbgt = recs(ind).objects(j).bbox; %[x y x+w y+h]
        ov1 = find_overlap_all(props_pos , bbgt);        
        ov_ov = false(length(ov1), length(IoU_rng));
        for o=1:length(IoU_rng)
            ov_ov(:,o) = ov1>=IoU_rng(o);
        end
        
        for xx=1:length(siz_win_all)
            ix_winsiz = (props_size == xx);            
            siz_hist_obj_ov(n,xx,1) = sum(ov_ov(:,1) & ix_winsiz); %0.5
            siz_hist_obj_ov(n,xx,2) = sum(ov_ov(:,2) & ix_winsiz); %0.6
            siz_hist_obj_ov(n,xx,3) = sum(ov_ov(:,3) & ix_winsiz); %0.7
            siz_hist_obj_ov(n,xx,4) = sum(ov_ov(:,4) & ix_winsiz); %0.8
            siz_hist_obj_ov(n,xx,5) = sum(ov_ov(:,5) & ix_winsiz); %0.9
        end             

        ov_all(n,1) = double(any(ov1 >= optwin.ov_thre));
        ov_all(n,2) = double(recs(ind).objects(j).difficult);
        ov_all(n,3:4) = [bbgt(4)-bbgt(2)+1, bbgt(3)-bbgt(1)+1]; 
        n=n+1;
    end
    %g_time = g_time + toc(gen_time);
    
    % display progress
    if toc>10
        cov_rate=sum(ov_all(:,1)==1)./size(ov_all,1);
        cov_rate_easy=sum(ov_all(:,1)==1 & ov_all(:,2)==0)./sum(ov_all(:,2)==0);
        fprintf('pr: load: %d/%d cover:%.2f cover_easy:%.2f',i, optwin.nimg_rand, cov_rate, cov_rate_easy);
        fprintf(' t:%0.4f  nprop:%5d\n', g_time/nn, round(np_m/nn));
        g_time=0; np_m=0; nn=0;
        drawnow; tic;
    end
    
    if rem(i,10)==0
       siz_hist_all = siz_hist_obj_ov(1:n,:,:);
       fname = sprintf(['siz_hist_all_ov_' 'VOC07' '.mat']);
       save(fname, 'siz_win_all', 'siz_hist_all', 'ov_all', '-v7.3');
    end
    
end

win_siz_selection();
