function [ov, ia] = find_overlap_all(bb, bbgt)
%bb:  [c r c+width r+height]
% bb=double(bb); bbgt=double(bbgt);
bi=[max(bb(:,1),bbgt(1)), max(bb(:,2),bbgt(2)), min(bb(:,3),bbgt(3)), min(bb(:,4),bbgt(4))];
iw=bi(:,3)-bi(:,1);
ih=bi(:,4)-bi(:,2);
cond = (iw>0 & ih>0);
% if                 
    % compute overlap as area of intersection / area of union
    ua=(bb(:,3)-bb(:,1)).*(bb(:,4)-bb(:,2))+...
       (bbgt(3)-bbgt(1)).*(bbgt(4)-bbgt(2))-...
       iw.*ih;
    ia = iw.*ih;
    ov=ia./ua;
% else

ia = ia .* double(cond);
ov = ov .* double(cond);
%     ia = 0;
%     ov = 0;
% end

end