function save_normalized(input_image, filename)
 
% function result = save_normalized(input_image, filename)
%
% normalize the values in input_image so that the minimum value is 
% 0 and the maximum value is 255, and save the normalized image 
% to the specified filename.

normalized = normalize_range(input_image); 
imwrite(uint8(normalized), filename);

 

