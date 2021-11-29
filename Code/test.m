%%
cd(code_directory)

s = filesep; % This gets the file separator character from the  system
test_faces_photos = strcat(training_directory, '\test_face_photos');
test_cropped_faces = strcat(training_directory, '\test_cropped_faces');
test_nonfaces = strcat(training_directory, '\test_nonfaces');
other_code = strcat(code_directory, '\given');
addpath([other_code s '00_common' s '00_detection'])
addpath([other_code s '00_common' s '00_images'])
addpath([other_code s '00_common' s '00_utilities'])
addpath(other_code)
addpath(test_faces_photos)
addpath(test_cropped_faces)
addpath(test_nonfaces)

load boosted_classifier
load weak_classifiers
load boosted_classifier_num
%%

number_faces = 5;

%read in the cropped faces (size 100 x 100)
face_images_cropped = dir(fullfile(test_cropped_faces,'*.bmp'));
cropped_faces = zeros(100, 100, number_faces);
for i = 1:size(face_images_cropped,3)
  filename = fullfile(test_cropped_faces,face_images_cropped(i).name);
  cropped_faces(:,:,i) = read_gray(filename);
end
%%
%Get an image and detect skin on it, return the matrix that's been
%thresholded

negative_histogram = read_double_image('negatives.bin');
positive_histogram = read_double_image('positives.bin');

face_images_photo = dir(fullfile(test_faces_photos,'*.jpg'));


for i = 1:size(face_images_photo, 1)
    
    %% Get skin pixels in logical values
    filename = fullfile(test_faces_photos,face_images_photo(i).name);
    color_photo_image = double(imread(filename));
    skin_prob_image = detect_skin(color_photo_image, positive_histogram,  negative_histogram);
    skin_prob_image = skin_prob_image > .90;
    
    inte = integral_image(skin_prob_image);
    %{
    skin_prob_image = imdilate(skin_prob_image, ones(2,2));
    [labels, number] = bwlabel(skin_prob_image, 4);
    counters = zeros(1,number);
    for n = 1:number
        % first, find all pixels having that label.
        component_image = (labels == n);
        % second, sum up all white pixels in component_image
        counters(n) = sum(component_image(:));
    end
    [area, id] = maxk(counters,1);  
    skin_logics = zeros(size(skin_prob_image,1),size(skin_prob_image,2));
    for n = 1:size(id,2)
        %skin_logics = skin_logics | (skin_prob_image & (labels(n) == id(n))); %this gets the hand
        skin_logics = skin_logics | labels == id(n);
    end
    figure();
    imshow(skin_logics,[]);
    %%
    %}
    %[result, boxes] = boosted_detector_demo(skin_prob_image, 1:1:1, boosted_classifier, weak_classifiers, [68,57], 2);
    
    [result, boxes] = boosted_detector_demo(skin_prob_image, 1:1:1, boosted_classifier, weak_classifiers, [68,57], 2);    
    
    figure();
    imshow(result, []);
end

