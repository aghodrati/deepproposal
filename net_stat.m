function nchannel = net_stat( net )

nchannel = zeros(1, length(net.layers));

for i=1:length(net.layers)
    t = net.layers{i}.type;
    if strcmp(t, 'conv') == 1        
        nchannel(i) = size(net.layers{i}.filters, 4);
    else
        nchannel(i) = nchannel(i-1);
    end


end

