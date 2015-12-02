function x_map = compute_featmaps(im, net, opts)

averageImage = single(net.normalization.averageImage);
im_avg = mean(mean(averageImage,1),2);

%fprintf('-pr: connecting to gpu ...\n');
gpu = gpuDevice(); %gpu
%get image statistics
[h, w, d] =  size(im);
if d==1, im=cat(3, im, im, im); end;
im_siz = [h,w];      
[~,smin] = min(im_siz);
im_s = single(im);
%extract feature maps
x_map = cell(length(opts.scales), length(net.layers));
for s=1:length(opts.scales)
    ss = opts.scales(s) ./ im_siz(smin);
    siz_scale = round(im_siz.*ss);
    im_ = imresize(im_s, siz_scale);
    im_ = im_ - repmat(im_avg, siz_scale);
    
    im_gpu_ = gpuArray(im_);
    res = vl_simplenn(net, im_gpu_);
    %res = vl_simplenn(net, im_); %cpu
    for lid=opts.layers
        x_map{s, lid} = gather(res(lid+1).x); %the first one is input
        %x_map{s, lid} = res(l).x; %cpu
    end
    clear res im_gpu_
    wait(gpu);    
    
end

end