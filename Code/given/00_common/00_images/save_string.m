function result = save_string(the_string, file_pointer)

% function result = save_string(the_string, file_pointer)

result = 0;
length = size(the_string, 2);

count = fwrite(file_pointer, length, 'int32');
if count ~= 1
    disp('save_string: failed to write length of string');
    return;
end

count = fwrite(file_pointer, the_string, 'char');
if count ~= length
    disp('save_string: failed to write all string characters');
    return;
end

result = 1;

