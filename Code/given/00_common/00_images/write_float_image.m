function result = write_float_image(matrix, filename)

% write_float_image(matrix, filename)
% writes a float image (three-dimensional array)
% in the format we use in the interface c++ program.

if ndims(matrix) == 2
    result = write_float_matrix(matrix, filename);
    return;
end

result = 0;
fp = fopen(filename, 'w');
if fp == -1
    disp(['failed to open ', filename]);
    return;
end

result = write_float_image2(matrix, fp);
fclose(fp);
