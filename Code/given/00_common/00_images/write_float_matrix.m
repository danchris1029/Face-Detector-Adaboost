function result = write_float_matrix(matrix, filename)

% write_float_matrix(matrix, filename)
% writes a double matrix in the format we use in the interface c++ program.

result = 0;
fp = fopen(filename, 'w');
if fp == -1
    disp(['failed to open ', filename]);
    return;
end

result = write_float_matrix2(matrix, fp);
fclose(fp);
