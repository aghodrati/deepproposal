function win_siz_selection()

nselect = 50;

load(['siz_hist_all_ov_' 'VOC07' '.mat'], 'siz_win_all', 'siz_hist_all')
hist00 = siz_hist_all;
hist0 = hist00;
hist0 = hist0>0;

AR_op = 1;

%%%AR objective: greedly select windows that at the same time cover AND fit to the ground-truth as much as possible 
if AR_op==1    
    %wght=repmat([1 1 1 1 1], size(hist0,1),1); %[.06 0.12 0.25 0.5 1]
    x = mean(hist0,3);
    prob = mean(x,1);
    [~,bb] = sort(prob,'descend');
    best  =[];
    best(1) = bb(1);
    for i=2:nselect %TODO %size(hist0,2)
        interBest = sum(hist0(:,best,:),2);
        interBest = interBest>0;
        interBest = permute(interBest,[1 3 2]);

        x = mean(mean(interBest,1));
        %x = interBest.*wght; x=mean(sum(x,2));
        r70 = mean(interBest(:,3));
        
        fprintf('Recall@.7=%3.2f  AR=%3.2f  nbox=%d\n', r70, x, length(best));
        count=1;
        for j=1:size(hist0,2)
            if ismember(bb(j),best(:))
                continue;
            end

            interBest = sum(hist0(:,[best bb(j)],:),2);
            interBest = interBest>0;
            interBest = permute(interBest,[1 3 2]);
            x = mean(mean(interBest,1));
            %x = interBest.*wght; x=mean(sum(x,2));
            
            if count==1
                best_xorr = x;   
                best_idx = j;
                count = 2;

            elseif x > best_xorr
                best_xorr = x;
                best_idx = j; 

            end        
        end

        best(i) = bb(best_idx);
    end

    %recall for IoU of [0.5 0.6 0.7 0.8 0.9]
    for ss=1:5
        win_ix=best(1:nselect);
        a=hist0(:,win_ix,ss)>0;
        all_obj=sum(a,2)>0;
        xx(ss)=sum(all_obj)./length(all_obj);
    end

    best_win = siz_win_all(best)';
    siz_win = best_win;
    fname = ['./best_siz_win_AR_' 'VOC07' '.mat'];
    save(fname,'siz_win');
end
