function result = read_spline(filename)

result = 0;
fid = fopen(filename, 'r');
if (fid <= 0)
  return
end

number = fscanf(fid, '%li');
xs = read_double_matrix2(fid);
ys = read_double_matrix2(fid);

result = [xs', ys'];

fclose(fid);  
