function result = read_long_matrix(filename)

% reads a long matrix in the format we use in the interface c++ program.

result = [];
fp = fopen(filename, 'r');
if fp == -1
    disp(['failed to open ', filename]);
    return;
end

result = read_long_matrix2(fp);
fclose(fp);
