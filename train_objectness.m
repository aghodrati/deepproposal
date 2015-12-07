function mdl_o = train_objectness( win_sizes, net_gpu, opts )
% if the models are already trained, it loads them otherwise train
% objecness models from scratch.

addpath(genpath('./deps/liblinear-2.1/'));

trainer.svm_C = 1;
trainer.bias_mult = 0;
trainer.pos_loss_weight = 1;
trainer.checkpoint = 200;
trainer.max_npos = 10;
trainer.max_nneg = 50;
trainer.pos_ov_thre = 0.7;
trainer.neg_ov_thre = 0.3;
trainer.nimg_rand = 5011;
trainer.layers_nchannel = net_stat(net_gpu);
opts.train = trainer;

mdl_o = cell(length(opts.scales), length(net_gpu.layers));
for layer_id = opts.layers(1:end-1) %last one is for refinement
    for scale_id=1:length(opts.scales)    
        opts.train.d = trainer.layers_nchannel(layer_id) + 3;
        opts.train.model_name = sprintf('%s/l%d/objectness_model_s%d.mat', opts.model.objectness, layer_id, opts.scales(scale_id));
        
        %trainer
        if ~exist(opts.train.model_name,'file')           
            mdl_o{scale_id,layer_id} = train_objectness_i(scale_id, layer_id, win_sizes{scale_id}, net_gpu, opts);
        else
            X = load(opts.train.model_name, 'objectness_model');
            mdl_o{scale_id,layer_id} = X.objectness_model;
        end
    end
end

        
        