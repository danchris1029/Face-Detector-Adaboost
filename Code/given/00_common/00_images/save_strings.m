function result = save_strings(strings, file_pointer)

% function result = save_strings(strings, file_pointer)

result = 0;
length = numel(strings);

count = fwrite(file_pointer, length, 'int32');
if count ~= 1
    disp('save_string: failed to write length of string');
    return;
end

for i=1:length
    the_string = strings{i};
    temp = save_string(the_string, file_pointer);
    if (temp <= 0)
        disp(sprintf('failed to save item %i', i));
        return;
    end
end

result = 1;

