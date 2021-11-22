function result = write_uchar_image(matrix, filename)

% write_integer_matrix(matrix, filename)
% writes a integer matrix in the format we use in the interface c++ program.

result = 0;
fp = fopen(filename, 'w');
if fp == -1
    disp(['failed to open ', filename]);
    return;
end

result = write_uchar_image2(matrix, fp);
fclose(fp);
