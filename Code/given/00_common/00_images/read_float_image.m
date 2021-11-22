function result = read_float_image(filename)

% reads a float image in the format we use in the interface c++ program.

result = [];
fp = fopen(filename, 'r');
if fp == -1
    disp(['failed to open ', filename]);
    return;
end

result = read_float_image2(fp);
fclose(fp);
