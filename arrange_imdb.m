function opts = arrange_imdb(opts)

data_path = ['./data/' opts.imdb.dname];    
if ~exist(data_path, 'dir'), mkdir(data_path); end;
    
%%%%%%%%%%%%%%%%%%%%%%%%
%VOC07
%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp(opts.imdb.dname, 'VOC07')
    
    vocdir = opts.imdb.dataset_path;
    opts.imdb.imgpath_trn = [vocdir '/VOC2007/JPEGImages/'];
    opts.imdb.imgpath_tst = opts.imdb.imgpath_trn;
    addpath(vocdir); addpath([vocdir 'VOCcode']);
    VOCinit;
    opts.imdb.trn_info_path = ['./data/' opts.imdb.dname '/all_trainval.mat'];
    opts.imdb.tst_info_path = ['./data/' opts.imdb.dname '/all_test.mat'];    
    opts.imdb.img_ext = 'jpg';    
    
    %trainval
    if ~exist(opts.imdb.trn_info_path,'file')
        voc_set = 'trainval';
        [gtids,t] = textread(sprintf(VOCopts.imgsetpath, voc_set),'%s %d');    
        recs = [];
        for i=1:length(gtids)
            rec = PASreadrecord(sprintf(VOCopts.annopath, gtids{i}));
            recs(i).imgsize = [rec.imgsize(1) rec.imgsize(2)];
            for o=1:length(rec.objects)
                recs(i).objects(o).bbox = rec.objects(o).bbox;
                recs(i).objects(o).boxlbl = rec.objects(o).class;
                recs(i).objects(o).difficult = rec.objects(o).difficult;
            end
        end
        save(opts.imdb.trn_info_path, 'recs', 'gtids');
    end
    
    %test
    voc_set = 'test';
    if ~exist(opts.imdb.tst_info_path,'file')
        [gtids,t] = textread(sprintf(VOCopts.imgsetpath, voc_set),'%s %d');    
        recs = [];
        for i=1:length(gtids)
            rec = PASreadrecord(sprintf(VOCopts.annopath, gtids{i}));
            recs(i).imgsize = [rec.imgsize(1) rec.imgsize(2)];
            for o=1:length(rec.objects)
                recs(i).objects(o).bbox = rec.objects(o).bbox;
                recs(i).objects(o).boxlbl = rec.objects(o).class;
                recs(i).objects(o).difficult = rec.objects(o).difficult;
            end
        end
        save(opts.imdb.tst_info_path, 'recs', 'gtids');
    end

%%%%%%%%%%%%%%%%%%%%%%%%
%COCO2014
%%%%%%%%%%%%%%%%%%%%%%%%
elseif strcmp(opts.imdb.dname, 'COCO14')
    
    coco_apidir = opts.imdb.dataset_apipath;
    coco_datadir = opts.imdb.dataset_path;
    opts.imdb.imgpath_trn = [coco_datadir '/images/train2014/'];
    opts.imdb.imgpath_tst = [coco_datadir '/images/val2014/'];    
    addpath(coco_apidir);
    opts.imdb.trn_info_path = ['./data/' opts.imdb.dname '/all_train.mat'];
    opts.imdb.tst_info_path = ['./data/' opts.imdb.dname '/all_val.mat'];    
    opts.imdb.img_ext = 'jpg';
    
    %train
    if ~exist(opts.imdb.trn_info_path,'file')
        dataType='train2014';
        annFile=sprintf('%s/annotations/instances_%s.json',coco_datadir,dataType);
        coco=CocoApi(annFile);
        imgIds = coco.getImgIds();    
        nimgs = length(imgIds);
        cats = coco.loadCats(coco.getCatIds());
        cat_mapper=[]; for i=1:length(cats), cat_mapper{cats(i).id} = cats(i).name; end;
        recs = []; gtids = cell(nimgs,1);
        tic;
        for i=1:nimgs
            if toc>10, fprintf('pr: collecting training annotations: %d/%d\n',i, nimgs); tic; end;
            imgId = imgIds(i);
            annId = coco.getAnnIds('imgIds',imgId, 'iscrowd',0);
            anns = coco.loadAnns(annId);
            imgInf = coco.loadImgs(imgId);
            gtids{i} = imgInf.file_name(1:end-4);
            recs(i).imgsize = [imgInf.width imgInf.height];
            for o=1:length(anns)
                b = round(anns(o).bbox);
                recs(i).objects(o).bbox = [b(1) b(2) b(1)+b(3)-1 b(2)+b(4)-1];
                recs(i).objects(o).boxlbl = cat_mapper(anns(o).category_id);
                recs(i).objects(o).difficult = 0;
                if (b(3)*b(4)<1024), recs(i).objects(o).difficult = 1; end; %TODO
            end
        end
        not_vld=[];
        for i=1:nimgs
            if isempty(recs(i).objects), not_vld = [not_vld, i]; end;
        end
        recs(not_vld) = []; gtids(not_vld)=[];
        save(opts.imdb.trn_info_path, 'recs', 'gtids');
    end
    
    %validation
    if ~exist(opts.imdb.tst_info_path,'file')
        dataType='val2014';
        annFile=sprintf('%s/annotations/instances_%s.json',coco_datadir,dataType);
        coco=CocoApi(annFile);
        imgIds = coco.getImgIds();    
        nimgs = length(imgIds);
        recs = []; gtids = cell(nimgs,1);
        cats = coco.loadCats(coco.getCatIds());
        cat_mapper=[]; for i=1:length(cats), cat_mapper{cats(i).id} = cats(i).name; end;
        tic;
        for i=1:length(imgIds)
            if toc>10, fprintf('pr: collecting validation annotations: %d/%d\n',i, nimgs); tic; end;
            imgId = imgIds(i);
            annId = coco.getAnnIds('imgIds',imgId, 'iscrowd',0);
            anns = coco.loadAnns(annId);
            imgInf = coco.loadImgs(imgId);
            gtids{i} = imgInf.file_name(1:end-4);
            recs(i).imgsize = [imgInf.width imgInf.height];
            for o=1:length(anns)
                b = round(anns(o).bbox);
                recs(i).objects(o).bbox = [b(1) b(2) b(1)+b(3)-1 b(2)+b(4)-1];
                recs(i).objects(o).boxlbl = cat_mapper(anns(o).category_id);
                recs(i).objects(o).difficult = 0;
                if (b(3)*b(4)<1024), recs(i).objects(o).difficult = 1; end; %TODO
            end
        end
        not_vld=[];
        for i=1:nimgs
            if isempty(recs(i).objects), not_vld = [not_vld, i]; end;
        end
        recs(not_vld) = []; gtids(not_vld)=[];
        save(opts.imdb.tst_info_path, 'recs', 'gtids');
    end  
end

