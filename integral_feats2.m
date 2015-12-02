
% ------------------------------------------------------------------------
function feats_intg = integral_feats2(x_feat_map, props)
% ------------------------------------------------------------------------

props(:, [3 4]) = props(:, [3 4]) + 1;

int_img = cumsum(cumsum(x_feat_map,2));
int_img = padarray(int_img,[1 1]);

y0 = reshape(permute(int_img,[3 1 2]), [size(int_img,3),size(int_img,1)*size(int_img,2)])';

ind1 = sub2ind([size(int_img,1) size(int_img,2)], props(:,4), props(:,3));
ind2 = sub2ind([size(int_img,1) size(int_img,2)], props(:,4), props(:,1));
ind3 = sub2ind([size(int_img,1) size(int_img,2)], props(:,2), props(:,3));
ind4 = sub2ind([size(int_img,1) size(int_img,2)], props(:,2), props(:,1));

% feats_intg = y0(ind1,:)-y0(ind2,:)-y0(ind3,:) + y0(ind4,:);
feats_intg = intg_sub(y0',single([ind1 ind2 ind3 ind4]))'; %mex implementation of previous line

%%box normalization
x = props(:,[3 4]) - props(:,[1 2]);
area = abs(x(:,1) .* x(:,2));

%method1
x_norm = sqrt(sum(feats_intg .^ 2, 2));  x_norm(x_norm < eps) = 1;  feats_intg = bsxfun(@rdivide, feats_intg, x_norm);

hyper_feats = [x area];
x_norm = sqrt(sum(hyper_feats .^ 2, 2)); x_norm(x_norm < eps) = 1; xa = bsxfun(@rdivide, hyper_feats, x_norm);

feats_intg = [feats_intg xa];