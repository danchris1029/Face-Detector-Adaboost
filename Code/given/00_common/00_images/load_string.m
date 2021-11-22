function result = load_string(file_pointer)

% function result = load_string(file_pointer)

result = [];

[length, number] = fscanf(file_pointer, '%i', 1);
if number < 1
    disp('load_string: failed to read length of string');
    return;
end

[result, number] = fscanf(file_pointer, '%s', 1);
if number ~= 1
    disp('load_string: failed to read all string characters');
    return;
end

result = char(result);
