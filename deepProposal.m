function [ props_all_nms ] = deepProposal( im1, x_map, mdl_o, mdl_e, win_sizes, opts )

    nscale = length(opts.scales);
    %get image size
    im_siz = [size(im1,1) size(im1,2)];
    %define variables to keep proposals in original/scaled image
    props_org_all=cell(nscale,1); 
    props_scl_all=cell(nscale,1);
    
    %%%%%%%%%%%%%%%%%%%%%
    %stage-1: conv5
    %%%%%%%%%%%%%%%%%%%%%      
    l = opts.layers(1); %conv5
    for s=1:nscale
        %generate boxes
        siz_box = [size(x_map{s,l},2) size(x_map{s,l},1)];
        [props_org, props_scl] = gen_props2(im_siz(2:-1:1), siz_box, win_sizes{s}); %[x y x+w y+h]
        %feature map for specific scale s
        x_feat_map = x_map{s,l};
        %generate integral features            
        feats_intg = integral_feats2(x_feat_map, props_scl);
        %scoring
        detectors = mdl_o{s,l}.detectors;        
        dec_values = bsxfun(@plus, feats_intg * detectors.W, detectors.B);
        dec_values_l = (dec_values-min(dec_values)) ./ (max(dec_values)-min(dec_values));
        scrs_ = dec_values_l;
        %nms
        props_nms = nms_c(single([props_org scrs_]), (opts.thr_s+opts.thr_e)/2 + 0.05 , opts.nbox_s1);
        [~,x1,x2] = intersect(props_org, props_nms(:,1:4),'rows'); [~,ixx] = sort(x2); pick=x1(ixx);
        props_org_all{s} = double([props_org(pick,:)  scrs_(pick,:)]);
        props_scl_all{s} = double(props_scl(pick,:));
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%
    %stage-2: conv3
    %%%%%%%%%%%%%%%%%%%%%
    
    for s=1:nscale
        %boxes from previous stage
        props_org = props_org_all{s}(:,1:4); 
        props_scl = props_scl_all{s};
        scrs_ = props_org_all{s}(:,5);
        for l = opts.layers(2:end-1)            
            %feature map for specific scale s
            x_feat_map = x_map{s,l};
            %generate integral image
            feats_intg = integral_feats2(x_feat_map, props_scl); %TODO
            %feats_intg = integral_feats_sp(x_feat_map, props_scl, 2);
            %scoring
            detectors = mdl_o{s,l}.detectors;
            dec_values = bsxfun(@plus, feats_intg * detectors.W, detectors.B);
            dec_values_l = (dec_values-min(dec_values)) ./ (max(dec_values)-min(dec_values));
            scrs_ = [scrs_ dec_values_l];
        end
        %score aggregation
        scrs = prod(scrs_, 2);
        %nms
        props_nms = nms_c(single([props_org scrs]), (opts.thr_s+opts.thr_e)/2 + 0.05, opts.nbox_s2);
        props_org_all{s} = double(props_nms);
        
    end
    
    %%proposal aggregation over different scales
    props_all_dmy = single(cell2mat(props_org_all));
    props_all_nms = nms_c(single(props_all_dmy), opts.thr_e, opts.nbox_s3, opts.step_siz);    
    
    %%%%%%%%%%%%%%%%%%%%%
    %stage-3: conv2
    %%%%%%%%%%%%%%%%%%%%%
    l = opts.layers(end); %conv2    
    fmap2 = imresize(x_map{s,l}, im_siz);
    nx = edgeBoxes_F(im1, props_all_nms(:,1:4), fmap2, mdl_e); 
    props_all_nms = [nx(:,1:4) props_all_nms(:,5)];
    
end

