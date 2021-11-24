% We're just using training images

% we don't find the best rectangle filter, but the best classifer when
% boosting

s = filesep; % This gets the file separator character from the  system
training_faces = strcat(training_directory, '\training_faces');
training_nonfaces = strcat(training_directory, '\training_nonfaces');
other_code = strcat(code_directory, '\given');
addpath([other_code s '00_common' s '00_detection'])
addpath([other_code s '00_common' s '00_images'])
addpath([other_code s '00_common' s '00_utilities'])
addpath(other_code)
addpath(training_faces)
addpath(training_nonfaces)

cd(code_directory)


best_boosted_classifer = zeros(0, 3);

%%

number_faces = 50;
number_nonfaces = 20;

face_images = dir(fullfile(training_faces,'*.bmp'));
nonface_images = dir(fullfile(training_nonfaces,'*.jpg'));

%nfiles = length(imagefiles);
faces = zeros(63, 57, number_faces);
for i = 1:number_faces
  filename = fullfile(training_faces,face_images(i).name);
  tempface = read_gray(filename);
  faces(:,:,i) = tempface(26:88, 22:78);
end

face_pool = zeros(63, 57, number_faces);
for i = 1:size(face_images, 1);
  filename = fullfile(training_faces,face_images(i).name);
  tempface = read_gray(filename);
  face_pool(:,:,i) = tempface(26:88, 22:78);
end
%figure(1)
%imshow(faces(:,:,27),[]);

nonfaces = zeros(size(faces,1), size(faces,2), number_nonfaces);
num_subwindows = 5;
non_faces_index = 1;
for i = 1:number_nonfaces
    filename = fullfile(training_nonfaces,nonface_images(i).name);
    temp_nonface = read_gray(filename);
    for j = 1:num_subwindows
        offset_x = floor((size(temp_nonface,1)-3*size(nonfaces,1)).*rand(1,1)+size(nonfaces,1));
        offset_y = floor((size(temp_nonface,2)-3*size(nonfaces,2)).*rand(1,1)+size(nonfaces,2));
        nonfaces(:,:,non_faces_index) = temp_nonface(19+offset_x:81+offset_x, 22+offset_y:78+offset_y);
        non_faces_index = non_faces_index+1;
    end
end
dimensions = [size(faces(:,:,1),1), size(faces(:,:,1),2)];%[100, 100];
% traing_face dimension is 100 x 100

% We have 5 thresholds per pixel
% so 10000 * 5 = 50000 classifers

%%

%dimensions = [1000, 1000];

% bootstrapping, adding wrong_faces into array of faces to increase
% strength of face detector on extreme cases

% create every weak classifer here in a for loop

%number = floor(dimensions(1) * dimensions(2) / 10);
number = 1000;
weak_classifiers = cell(1, 0);

rounds = 20;
for round = 1: rounds

    %disp("round ", round);
    %number = (dimensions(1) * dimensions(2)) / 10;
    %weak_classifiers = cell(1, number);
    temp_classifer = cell(1, number);
    
    for i = 1:number
        temp_classifer{i} = generate_classifier(dimensions(1), dimensions(2));
    end

    weak_classifiers = cat(2, weak_classifiers, temp_classifer); 
    
    % save classifers into temp variable
    
    
    %{
    example_number = size(faces, 3) + size(nonfaces, 3);
    labels = zeros(example_number, 1);
    labels (1:size(faces, 3)) = 1;
    labels((size(faces, 3)+1):example_number) = -1;
    examples = zeros(face_vertical, face_horizontal, example_number);
    examples (:, :, 1:size(faces, 3)) = face_integrals;
    examples(:, :, (size(faces, 3)+1):example_number) = nonface_integrals;
    %}
    face_integrals = zeros(size(faces,1), size(faces,2), number_faces);
    for i = 1:size(faces, 3)
        face_integrals(:,:,i) = integral_image(faces(:, :, i));
    end
    nonface_integrals = zeros(size(faces,1), size(faces,2), number_nonfaces);
    for i = 1:size(nonfaces, 3)
        nonface_integrals(:,:,i) = integral_image(nonfaces(:, :, i));
    end

    example_number = size(faces, 3) + size(nonfaces, 3);
    labels = zeros(example_number, 1);
    labels (1:size(faces, 3)) = 1;
    labels((size(faces, 3)+1):example_number) = -1;
    
    examples = zeros(dimensions(1), dimensions(2), example_number);
    examples (:, :, 1:size(faces, 3)) = face_integrals;
    examples(:, :, (size(faces, 3)+1):example_number) = nonface_integrals;

    classifier_number = numel(weak_classifiers);

    responses = zeros(classifier_number, example_number);

    for i = 1:example_number
        %integral = examples(:, :, example);
        integral = examples(:, :, i);
        for feature = 1:classifier_number
            classifier = weak_classifiers {feature};
            responses(feature, i) = eval_weak_classifier(classifier, integral);
        end
        %disp(i)
    end

    % sum the responses from each image for each classifier together into one
    % row within responses
    boosted_classifier_num = 5;
    boosted_classifier = AdaBoost(responses, labels, boosted_classifier_num);

    num_wrong_face = 0;

    for i = 1:size(face_pool,3)
        prediction = boosted_predict(face_pool(:, :, i), boosted_classifier, weak_classifiers, boosted_classifier_num);
        if (prediction < 0)
           num_wrong_face = num_wrong_face + 1; 
           faces = cat(3, face_pool(:, :, i), faces);
        end
    end
    
    
    % We need a non_face pool for false negatives
    %{
    num_wrong_nonface = 0;

    for i = 1:size(nonfaces,3)
        prediction = boosted_predict(nonfaces(:, :, i), boosted_classifier, weak_classifiers, boosted_classifier_num);
        if (prediction > 0)
            num_wrong_nonface = num_wrong_nonface + 1;
        end
    end
    %}

    if(num_wrong_face == 0)
        break;
    end
end

save boosted_classifier boosted_classifier
save weak_classifiers weak_classifiers
save boosted_classifier_num boosted_classifier_num

%%
%saved_boosetd_classifier = boosted_classifier;
%boosted_classifier = saved_boosetd_classifier;

% repeat training N times and keep only the classifers with positive alphas, and
% replace the classifers with negative alphas, with new weak classifers
%%
%{
for i = 1: 5
    best_boosted_classifer_index = find(max(boosted_classifier(:, 2)) == boosted_classifier(:, 2), 1);
    best_boosted_classifer(i, :) = boosted_classifier(best_boosted_classifer_index, :);
    
    best_boosted_classifer = cat(1, best_boosted_classifer, boosted_classifier(best_boosted_classifer_index, :));
    
    boosted_classifier(best_boosted_classifer_index, :) = -inf;
    
end
%}
%%

%kept_classifiers = cell(1, boosted_classifier_num);
%for i = 1:boosted_classifier_num
%    kept_classifiers{i} = weak_classifiers{boosted_classifier(i,1)};
%end

%%
%{
num_wrong_nonface = 0;

for i = 1:size(nonfaces,3)
    prediction = boosted_predict(nonfaces(:, :, i), boosted_classifier, weak_classifiers, boosted_classifier_num);
    if (prediction > 0)
        num_wrong_nonface = num_wrong_nonface + 1;
    end
end
%}
% Each time we create a weak classifer, we use generate_classifer

% Every weak classifer has soft classifer per threshold

% Top 1000 responses will become the  classifer

% get integral skin image


% generate classifer