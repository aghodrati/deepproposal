function [wndw, nwin] = voc07_conv_locs(siz_box, siz_win, stride)
%siz_box is in [w h] format

if nargin<3, stride=32; end;
if nargin<2, siz_win=[227 227]; end;

w_win=siz_win(1);
h_win=siz_win(2);

%[h, w] = size(box);
h=siz_box(2); w=siz_box(1);
nwin_x = floor((w-w_win)./stride)+1;
nwin_y = floor((h-h_win)./stride)+1;
n=1;
if (nwin_x<1 || nwin_y<1), wndw=[]; nwin=[0 0]; return; end;

wndw = zeros(nwin_y*nwin_x,4);
for x=1:stride:nwin_x*stride
    for y=1:stride:nwin_y*stride
        wndw(n,:) = [x y w_win h_win];
        n=n+1;
    end
end
nwin = [nwin_x nwin_y];