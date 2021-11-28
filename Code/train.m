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
for i = 1:size(face_images, 1)
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

nonface_pool = zeros(size(faces,1), size(faces,2), size(nonface_images,1) * num_subwindows);
for i = 1:size(nonface_images,1)
    filename = fullfile(training_nonfaces,nonface_images(i).name);
    temp_nonface = read_gray(filename);
    for j = 1:num_subwindows
        offset_x = floor((size(temp_nonface,1)-3*size(nonface_pool,1)).*rand(1,1)+size(nonface_pool,1));
        offset_y = floor((size(temp_nonface,2)-3*size(nonface_pool,2)).*rand(1,1)+size(nonface_pool,2));
        nonface_pool(:,:,non_faces_index) = temp_nonface(19+offset_x:81+offset_x, 22+offset_y:78+offset_y);
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

boosted_classifier = bootstrapping_adaboost(dimensions,faces,nonfaces,face_pool,nonface_pool,number_faces,number_nonfaces);


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