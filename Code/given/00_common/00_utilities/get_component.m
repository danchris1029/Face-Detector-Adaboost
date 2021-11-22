function result = get_component(binary, k)

%  result = get_component(binary)
%
% returns a boolean image where non-zero pixels represent
% the k-th largest connected component of the input image

% connected component analysis
[labels, number] = bwlabel(binary, 4);
if number < k
    result = [];
    return;
end

[rows, cols] = size(binary);

% find the largest connected component
% create an array of counters, one for each connected component.
counters = zeros(1,number);
for row = 1:rows
    for col = 1:cols
        % for each i ~= 0, we count the number of pixels equal to i in the labels
        % matrix
        % first, we create a component image, that is 1 for pixels belonging to
        % the i-th connected component, and 0 everywhere else.

        label = labels(row, col);
        if (label ~= 0)
            counters(label) = counters(label) + 1;
        end
    end
end
    
% find the id of the largest component
% we pass -counters as argument because we want to find max areas,
% not min areas.
[areas, ids] = sort(-counters);    
result = (labels == ids(k));
