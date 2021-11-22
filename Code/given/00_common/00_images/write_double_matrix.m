function result = write_double_matrix(matrix, filename)

% writes a double matrix in the format we use in the interface c++ program.

result = 0;
fp = fopen(filename, 'w');
if fp == -1
    disp(['failed to open ', filename]);
    return;
end

result = write_double_matrix2(matrix, fp);
fclose(fp);
