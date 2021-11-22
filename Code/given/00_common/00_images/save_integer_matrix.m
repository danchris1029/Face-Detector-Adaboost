function result = save_integer_matrix(matrix, filename)

% save_integer_matrix(matrix, filename)
% writes a integer matrix in the format we use in the interface c++ program.
% speech-recognition friendly wrapper for write_integer_matrix

result = write_integer_matrix(matrix, filename);
return;

