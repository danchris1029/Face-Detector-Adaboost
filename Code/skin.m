

% read histograms
negative_histogram = read_double_image('negatives.bin');
positive_histogram = read_double_image('positives.bin');


image = double(imread('v1b.bmp'));
if(size(image, 3) == 3)
    result_on_skin = detect_skin(image, positive_histogram,  negative_histogram);
  
    skin_threshold = 0.6;
    skin_pixels = result_on_skin > skin_threshold;
    figure (1); imshow(result_on_skin, []);
    
    scale = 0.5;
    scaled_image = imresize(image, 1/double(scale), 'bilinear');
    scaled_skin_pixels = imresize(skin_pixels, 1/double(scale), 'bilinear');

    scaled_skin_integral = integral_image(skaled_skin_pixels);
    % Specify window offset (top-left corner)
top_offset = 700; 
left_offset = 1050;

% Specify window size 
wrows = 350;
wcols = 300;

%%
% Counting the skin pixels without the use of integral image
tic;
window = scaled_skin_pixels(top_offset:top_offset+wrows-1, left_offset:left_offset+wcols-1);
sum_skin_pixels = sum(sum(window))
toc;

%%
% Counting the skin pixels with the use of integral image
tic;
area1 = scaled_skin_integral(top_offset - 1, left_offset - 1);
area2 = scaled_skin_integral(top_offset + wrows - 1, left_offset + wcols - 1);
area3 = scaled_skin_integral(top_offset + wrows - 1, left_offset - 1);
area4 = scaled_skin_integral(top_offset - 1, left_offset + wcols - 1);

result = area1 + area2 - area3 - area4
toc;

% Note: counting skin pixels using the integral skin image should be faster
% than counting the skin pixels on the original skin detection image,
% especially for large window sizes. However, times measured in practice
% may vary due to a number of factors.

end
      
