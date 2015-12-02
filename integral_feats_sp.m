
% ------------------------------------------------------------------------
function feats_intg = integral_feats_sp(x_feat_map, props, ngrid)
% ------------------------------------------------------------------------
if nargin<3, ngrid=4; end;


int_img = cumsum(cumsum(x_feat_map,2));
int_img = padarray(int_img,[1 1]);


y0 = reshape(permute(int_img,[3 1 2]), [size(int_img,3),size(int_img,1)*size(int_img,2)])';

img_siz = [size(int_img,1), size(int_img,2)];


props_l{1} = [props(:,1) props(:,2)  props(:,3)  props(:,4)];

x_c = props(:,1)-1+max(round((props(:,3)-props(:,1)+1)/2),1);
y_c = props(:,2)-1+max(round((props(:,4)-props(:,2)+1)/2),1);
props_l{2} = [props(:,1) props(:,2) x_c y_c];
props_l{3} = [x_c+1 props(:,2) props(:,3) y_c];
props_l{4} = [props(:,1) y_c+1 x_c props(:,4)];
props_l{5} = [x_c+1 y_c+1 props(:,3) props(:,4)];

if ngrid==4
    x_cc1 = props(:,1)-1+max(round((x_c-props(:,1)+1)/2),1);
    y_cc1 = props(:,2)-1+max(round((y_c-props(:,2)+1)/2),1);
    x_cc2 = x_c-1+max(round((props(:,3)-x_c+1)/2),1);
    y_cc2 = y_c-1+max(round((props(:,4)-y_c+1)/2),1);

    props_l{6} = [props(:,1) props(:,2) x_cc1 y_cc1];
    props_l{7} = [x_cc1+1 props(:,2) x_c y_cc1];
    props_l{8} = [x_c+1 props(:,2) x_cc2 y_cc1];
    props_l{9} = [x_cc2+1 props(:,2) props(:,3) y_cc1];

    props_l{10} = [props(:,1) y_cc1+1 x_cc1 y_c];
    props_l{11} = [x_cc1+1 y_cc1+1 x_c y_c];
    props_l{12} = [x_c+1 y_cc1+1 x_cc2 y_c];
    props_l{13} = [x_cc2+1 y_cc1+1 props(:,3) y_c];

    props_l{14} = [props(:,1) y_c+1 x_cc1 y_cc2];
    props_l{15} = [x_cc1+1 y_c+1 x_c y_cc2];
    props_l{16} = [x_c+1 y_c+1 x_cc2 y_cc2];
    props_l{17} = [x_cc2+1 y_c+1 props(:,3) y_cc2];

    props_l{18} = [props(:,1) y_cc2+1 x_cc1 props(:,4)];
    props_l{19} = [x_cc1+1 y_cc2+1 x_c props(:,4)];
    props_l{20} = [x_c+1 y_cc2+1 x_cc2 props(:,4)];
    props_l{21} = [x_cc2+1 y_cc2+1 props(:,3) props(:,4)];
end

d=size(y0,2);
feats_intg=[]; %zeros(size(props,1), ngrid.^2*d);
for i=1:length(props_l)
    [ind_1, ind_2, ind_3, ind_4] = get_ix_int_img(props_l{i}, img_siz);
    %feats_l = y0(ind_1,:)-y0(ind_2,:)-y0(ind_3,:) + y0(ind_4,:);
    feats_l = intg_sub(y0',single([ind_1 ind_3 ind_2 ind_4]))';    
    x_norm = sqrt(sum(feats_l .^ 2, 2)); x_norm(x_norm < eps) = 1; feats_l = bsxfun(@rdivide, feats_l, x_norm);
    feats_intg=cat(2,feats_intg, feats_l);
    %feats_intg(:,(i-1)*d+1:i*d) = feats_l;
end

 
x = props(:,[3 4]) - props(:,[1 2]); area = abs(x(:,1) .* x(:,2));
x_norm = sqrt(sum([x area] .^ 2, 2)); x_norm(x_norm < eps) = 1; xa = bsxfun(@rdivide, [x area], x_norm);
feats_intg = [feats_intg xa];



function [ind_rb, ind_rt, ind_lb, ind4_lt] = get_ix_int_img(props, img_siz)

props(:, [3 4]) = props(:, [3 4]) + 1;

ind_rb = sub2ind([img_siz(1) img_siz(2)], props(:,4), props(:,3));
ind_rt = sub2ind([img_siz(1) img_siz(2)], props(:,4), props(:,1));
ind_lb = sub2ind([img_siz(1) img_siz(2)], props(:,2), props(:,3));
ind4_lt = sub2ind([img_siz(1) img_siz(2)], props(:,2), props(:,1));

