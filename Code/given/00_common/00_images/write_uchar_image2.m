function result = write_uchar_image2(matrix, fp)

% same as write_integer_matrix, but here we pass in a file pointer

result = 0;

rows = size(matrix, 1);
cols = size(matrix, 2);
bands = size(matrix, 3);
count = fwrite(fp, [6, rows, cols, bands], 'int32');

if count ~= 4
    disp('failed to write header');
    return;
end

for band = 1:bands
    mat = matrix(:,:,band);

    count = fwrite(fp, mat', 'uint8');

    if count ~= rows * cols
        disp(sprintf('failed to write data, count = %li', count));
        return;
    end
end


result = 1;
