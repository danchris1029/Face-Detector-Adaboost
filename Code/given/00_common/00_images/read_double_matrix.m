function result = read_double_matrix(filename)

% reads a float matrix in the format we use in the interface c++ program.

result = [];
fp = fopen(filename, 'r');
if fp == -1
    disp(['failed to open ', filename]);
    return;
end

result = read_double_matrix2(fp);
fclose(fp);
