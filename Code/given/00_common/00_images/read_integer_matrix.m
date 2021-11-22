function result = read_integer_matrix(filename)

% result = read_integer_matrix(filename)
% reads a 32-bit integer matrix in the format we use in the interface c++ program.

result = [];
fp = fopen(filename, 'r');
if fp == -1
    disp(['failed to open ', filename]);
    return;
end

result = read_integer_matrix2(fp);
fclose(fp);
