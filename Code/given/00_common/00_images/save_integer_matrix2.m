function result = save_integer_matrix2(matrix, file_pointer)

% save_integer_matrix2(matrix, filename)
% writes a integer matrix in the format we use in the interface c++ program.
% writes to an open file handle
% speech-recognition friendly wrapper for write_integer_matrix2

result = write_integer_matrix2(matrix, file_pointer);
return;

