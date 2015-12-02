%the code published as a part of https://github.com/hosang/detection-proposals
function [overlap, recall, AR] = compute_average_recall(unsorted_overlaps, dx)
  all_overlaps = sort(unsorted_overlaps(:)', 'ascend');
  num_pos = numel(all_overlaps);
  if nargin<2, dx = 0.1; end;

  overlap = 0:dx:1;
  overlap(end) = 1;
  recall = zeros(length(overlap), 1);
  for i = 1:length(overlap)
    recall(i) = sum(all_overlaps >= overlap(i)) / num_pos;
  end

  good_recall = recall(overlap >= 0.5);
  AR = 2 * dx * trapz(good_recall);
end
