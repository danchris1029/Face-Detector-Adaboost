function result = read_float3_matrix2(fp)

% same as read_float3_matrix, but here we pass in a file pointer

[header, count] = fread(fp, 4, 'int32');
result = [];

if count ~= 4
    disp('failed to read header');
    return;
elseif header(1) ~= 4
    disp(sprintf('bad entry in header 1: %li', header(1)));
    return;
elseif header(4) <= 0
    disp(sprintf('bad number of bands: %li', header(4)));
    return;
end

rows = header(2);
cols = header(3);
bands = header(4);
success = 1;

result = zeros(rows, cols, bands);
for band = 1:bands
    [temp, count] = fread(fp, [cols, rows], 'float32');
    if count ~= rows * cols
        disp(sprintf('failed to read data, count = %li', count));
        success = 0;
    end
    result(:, :, band) = temp';
end

if success == 0
    result = 0;
end
