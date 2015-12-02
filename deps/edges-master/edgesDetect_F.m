function [E,O,inds,segs] = edgesDetect_F( I, model, feat)
% Detect edges in image.
%
% For an introductory tutorial please see edgesDemo.m.
%
% The following model params may be altered prior to detecting edges:
%  prm = stride, sharpen, multiscale, nTreesEval, nThreads, nms
% Simply alter model.opts.prm. For example, set model.opts.nms=1 to enable
% non-maximum suppression. See edgesTrain for parameter details.
%
% USAGE
%  [E,O,inds,segs] = edgesDetect( I, model )
%
% INPUTS
%  I          - [h x w x 3] color input image
%  model      - structured edge model trained with edgesTrain
%
% OUTPUTS
%  E          - [h x w] edge probability map
%  O          - [h x w] coarse edge normal orientation (0=left, pi/2=up)
%  inds       - [h/s x w/s x nTreesEval] leaf node indices
%  segs       - [g x g x h/s x w/s x nTreesEval] local segmentations
%
% EXAMPLE
%
% See also edgesDemo, edgesTrain, edgesChns
%
% Structured Edge Detection Toolbox      Version 3.01
% Code written by Piotr Dollar, 2014.
% Licensed under the MSR-LA Full Rights License [see license.txt]

% get parameters
opts=model.opts; opts.nTreesEval=min(opts.nTreesEval,opts.nTrees);
if(~isfield(opts,'sharpen')), opts.sharpen=0; end
if(~isfield(model,'segs')), model.segs=[]; model.nSegs=[]; end
opts.stride=max(opts.stride,opts.shrink); model.opts=opts;

% pad image, making divisible by 4
siz=size(I); r=opts.imWidth/2; p=[r r r r];
p([2 4])=p([2 4])+mod(4-mod(siz(1:2)+2*r,4),4);
I = imPad(I,p,'symmetric');

% compute features and apply forest to image
%   [chnsReg,chnsSim] = edgesChns( I, opts );
chnsReg = imPad(feat,p,'symmetric');
chnsSim = [];
s=opts.sharpen; if(s), I=convTri(rgbConvert(I,'rgb'),1); end
[E,inds] = edgesDetectMex(model,I,chnsReg,chnsSim);

% normalize and finalize edge maps
t=opts.stride^2/opts.gtWidth^2/opts.nTreesEval; r=opts.gtWidth/2;
if(s==0), t=t*2; elseif(s==1), t=t*1.8; else t=t*1.66; end
E=E(1+r:siz(1)+r,1+r:siz(2)+r,:)*t; E=convTri(E,1);

% compute approximate orientation O from edges E
if( opts.nms==-1 ), O=[]; elseif( nargout>1 || opts.nms )
  [Ox,Oy]=gradient2(convTri(E,4));
  [Oxx,~]=gradient2(Ox); [Oxy,Oyy]=gradient2(Oy);
  O=mod(atan(Oyy.*sign(-Oxy)./(Oxx+1e-5)),pi);
end

% perform nms
if( opts.nms>0 ), E=edgesNmsMex(E,O,1,5,1.01,opts.nThreads); end

end
