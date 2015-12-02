
%%% generate hypothesis bboxes
function [bbox_pred, win_on, selected_win] = gen_props3(im_siz, siz_box, siz_wins)
%im_siz and siz_box format: [w h]
%bbox_pred format: [x y x+w y+h]


stride=1;
selected_win=[];
win_on=zeros(100000,5); nw=0;
for ii=1:length(siz_wins)
    siz_win=siz_wins{ii};
    [wndw, ~] = voc07_conv_locs(siz_box, siz_win, stride); %wndw:[c r w h]
    if isempty(wndw) continue; end
    wndw(:,5) = ii;
    win_on(nw+1:nw+size(wndw,1), :) = wndw; nw=nw+size(wndw,1);    
end
win_on(nw+1:end,:) = [];

%%% transform hypotheses to source (original image);
siz_scale = im_siz./siz_box;
win_on2=win_on;
win_on2(:,[1 2]) = win_on2(:,[1 2])-1;
bbox_pred = zeros(size(win_on2,1),4);

bbox_pred(:,[1 3]) = win_on2(:,[1 3]) * siz_scale(1);
bbox_pred(:,[2 4]) = win_on2(:,[2 4]) * siz_scale(2);
x=round(bbox_pred); bbox_pred = [max(1, x(:,[1 2])) , min(im_siz(1), x(:,3)) , min(im_siz(2), x(:,4)) ];
% bbox_pred = [ceil(bbox_pred(:,1:2))+1 floor(bbox_pred(:,3:4))];
bbox_pred(:,3:4) = bbox_pred(:,1:2) + bbox_pred(:,3:4)-1;
bbox_pred(:,5) = win_on(:,5);
win_on(:,[3 4]) = win_on(:,[3 4]) + win_on(:,[1 2]) - 1;

end

