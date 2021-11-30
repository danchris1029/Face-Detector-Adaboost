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
load boosted_classes

%%
%Get an image and detect skin on it, return the matrix that's been
%thresholded

negative_histogram = read_double_image('negatives.bin');
positive_histogram = read_double_image('positives.bin');

face_images_photo = dir(fullfile(test_faces_photos,'*.jpg'));

%for i = 1:size(face_images_photo, 1)
%% Get skin pixels in logical values
    face_size = [63, 57];
    
    image_index = 1;
    
    filename = fullfile(test_faces_photos,face_images_photo(image_index).name);
    color_photo_image = double(imread(filename));
    skin_prob_image = detect_skin(color_photo_image, positive_histogram,  negative_histogram);
    skin_prob_image = skin_prob_image > .7;
   
    %predicted_faces = zeros(0, 3);
    
    test_gray = read_gray(filename);
    test_skin_gray = (test_gray .*(skin_prob_image));
    max_responses = zeros(size(test_gray, 1), size(test_gray, 2));

    for scale = 1:.5:2
    
    scaled_image = imresize(test_skin_gray, scale, 'bilinear');
    
    %scaled_width = ceil(63 * (1/scale));
    %scaled_height = ceil(57 * (1/scale));
	for i = 1: size(scaled_image,1) - 63
        for j = 1: size(scaled_image,2) - 57	
            %i = 117;
            %j = 152;
            %i
            subwindow = scaled_image(i:(i+63-1), j:(j+57-1));
            
            for classifier_index = 1: size(boosted_classes, 2)
                
                boosted_model = boosted_classes{classifier_index};
                prediction = boosted_predict(subwindow, boosted_model, weak_classifiers, size(boosted_model, 1));
                %prediction
                
                if(prediction < 0)
                    classifier_index = 1;
                    break;
                end
            end
            %classifer_index
            if (classifier_index == size(boosted_classes, 2))
                %disp("here");
                response = prediction;
                y = round(i / scale);
                x = round(j / scale);
                
                max_responses(y, x) = max_responses(y, x) + response;
                
                %predicted_faces = cat(1, predicted_faces, [response, y, x]);
            end
        end 
    end
    scale
    end
   
    result_number = 3;
    boxes = detection_boxes(skin_prob_image, zeros(face_size), max_responses, ...
                         ones(size(test_gray, 1), size(test_gray, 2)), result_number);
    
                     
    result = test_gray;                 
    for number = 1: result_number
        result = draw_rectangle1(result, boxes(number, 1), boxes(number, 2), ...
                             boxes(number, 3), boxes(number, 4));
    end
   figure();
   imshow(result, []);    
    
    %{
    [result, boxes] = boosted_detector_demo(skin_prob_image, 1:1:1, boosted_classifier, weak_classifiers, [63,57], 5);    
    
    figure();
    imshow(result, []);
%end
   
  %}



%read in the cropped faces (size 100 x 100)
face_images_cropped = dir(fullfile(test_cropped_faces,'*.bmp'));
false_negative_faces = 0;
detected_faces = 0;
cropped_faces = zeros(63,57, size(face_images_cropped,1));
for i = 1:size(face_images_cropped,1)
    filename = fullfile(test_cropped_faces,face_images_cropped(i).name);
    tempface = read_gray(filename);
    cropped_faces(:,:,i) = tempface(26:88, 22:78);
    result = boosted_predict(cropped_faces(:,:,i), boosted_classifier, weak_classifiers);

    if result > 0 
        detected_faces = detected_faces +1;
    else
        false_negative_faces = false_negative_faces +1;
    end
end
false_detection_rate_faces = false_negative_faces/size(face_images_cropped,1);

nonfaces_test = dir(fullfile(test_nonfaces,'*.bmp'));
false_negative_nonfaces = 0;
detected_nonfaces = 0;
nonfaces = zeros(63,57, size(nonfaces_test,1));
for i = 1:size(nonfaces_test,1)
    filename = fullfile(test_nonfaces,nonfaces_test(i).name);
    tempface = read_gray(filename);
    nonfaces(:,:,i) = reshape(tempface, [63,57]);
    result = boosted_predict(nonfaces(:,:,i), boosted_classifier, weak_classifiers);

    if result > 0 
        detected_nonfaces = detected_nonfaces +1;
    else
        false_negative_nonfaces = false_negative_nonfaces +1;
    end
end
false_detection_rate_nonfaces = false_negative_nonfaces/size(nonfaces_test,1);

