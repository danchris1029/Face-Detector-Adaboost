function result = read_entry(filename, channel, vertical, horizontal)

% function result = read_entry(filename)
%
% assuming that a 3D matrix is stored (using my format) at the specified
% filename, this function returns the entry corresponding to the specified
% 3D coordinates.l

result = 0;

if nargin ~= 4 || nargout > 1
    disp('usage: result = read_entry(filename)');
    return;
end


file_pointer = fopen(filename);
if file_pointer <= 0
    disp(['failed to open ', filename]);
    return;
end

header = read_matrix_header(file_pointer);

entry_type = header(1);

if entry_type == 4   % float type
    entry_string = 'float32';
    entry_size = 4;
else
    disp(sprintf('cannot handle entry type %f', entry_type));
    fclose(file_pointer);
    return;
end

channels = header(4);
vertical_size = header(2);
horizontal_size = header(3);

displacement = entry_size * (channel * vertical_size * horizontal_size + vertical * horizontal_size + horizontal);
error_code = fseek(file_pointer, displacement, 'cof');
if error_code ~= 0
    disp(sprintf('failed to fseek to displacement %f', displacement));
    fclose(file_pointer);
    return;
end

[result, items] = fread(file_pointer, 1, entry_string);
fclose(file_pointer);

if items ~= 1
    disp('failed to read entry from disk');
end
