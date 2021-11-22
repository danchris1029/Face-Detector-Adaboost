function result = write_float_image2(matrix, fp)

% same as write_image_matrix, but here we pass in a file pointer

if ndims(matrix) == 2
    result = write_float_matrix2(matrix, fp);
    return;
end

result = 0;

channels = size(matrix, 1);
vertical = size(matrix, 2);
horizontal = size(matrix, 3);

count = fwrite(fp, [4, vertical, horizontal, channels], 'int32');
if count ~= 4
    disp('failed to write header');
    return;
end

for counter = 1: channels
    temporary = matrix(counter,:,:);
    temporary = reshape(temporary, vertical, horizontal);
    temporary = temporary';
    count = fwrite(fp, temporary, 'float32');
    if count ~= vertical * horizontal
        disp(sprintf('failed to write data, count = %li', count));
        return;
    end
end

result = 1;
