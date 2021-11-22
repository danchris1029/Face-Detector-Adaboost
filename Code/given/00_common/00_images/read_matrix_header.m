function result = read_matrix_header(file_pointer)

% function result = read_matrix_header(file_pointer)


[header, count] = fread(file_pointer, 4, 'int32');
result = [];

if count ~= 4
    disp('failed to read header');
    return;
end

result = header;

