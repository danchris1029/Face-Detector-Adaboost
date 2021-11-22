function result = read_matrix_header2(filename)

% function result = read_matrix_header2(filename)

file_pointer = fopen(filename);
if file_pointer <= 0
    disp(['failed to open ', filename]);
    return;
end

result = read_matrix_header(file_pointer);
