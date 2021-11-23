% We're just using training images

% we don't find the best rectangle filter, but the best classifer when
% boosting

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
number_faces = 300;
number_nonfaces = 50;

face_images = dir(fullfile(training_faces,'*.bmp'));
nonface_images = dir(fullfile(training_nonfaces,'*.jpg'));

%nfiles = length(imagefiles);
faces = zeros(100, 100, number_faces);
for i = 1:number_faces
  filename = fullfile(training_faces,face_images(i).name);
  faces(:,:,i)=read_gray(filename);
end

nonfaces = zeros(100, 100, number_nonfaces);
for i = 1:number_nonfaces
    filename = fullfile(training_nonfaces,nonface_images(i).name);
    temp_nonface = read_gray(filename);
    nonfaces(:,:,i) = imresize(temp_nonface,[100,100]);
end
%imshow(nonfaces(:,:,5),[]);
dimensions = [size(faces(:,:,1),1), size(faces(:,:,1),2)];%[100, 100];
% traing_face dimension is 100 x 100

% We have 5 thresholds per pixel
% so 10000 * 5 = 50000 classifers

number = (dimensions(1) * dimensions(2))/10;
weak_classifiers = cell(1, number);
for i = 1:number
    weak_classifiers{i} = generate_classifier(dimensions(1), dimensions(2));
end

%{
example_number = size(faces, 3) + size(nonfaces, 3);
labels = zeros(example_number, 1);
labels (1:size(faces, 3)) = 1;
labels((size(faces, 3)+1):example_number) = -1;
examples = zeros(face_vertical, face_horizontal, example_number);
examples (:, :, 1:size(faces, 3)) = face_integrals;
examples(:, :, (size(faces, 3)+1):example_number) = nonface_integrals;
%}
face_integrals = zeros(100, 100, number_faces);
for i = 1:size(faces, 3)
    face_integrals(:,:,i) = integral_image(faces(:, :, i));
end
nonface_integrals = zeros(100, 100, number_nonfaces);
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
boosted_classifier_num = 15;
boosted_classifier = AdaBoost(responses, labels, boosted_classifier_num);

%kept_classifiers = cell(1, boosted_classifier_num);
%for i = 1:boosted_classifier_num
%    kept_classifiers{i} = weak_classifiers{boosted_classifier(i,1)};
%end

save boosted_classifier boosted_classifier
save weak_classifiers weak_classifiers
save boosted_classifier_num boosted_classifier_num

num_wrong_face = 0;
num_wrong_nonface = 0;
for i = 1:size(faces,3)
    prediction = boosted_predict(faces(:, :, i), boosted_classifier, weak_classifiers, boosted_classifier_num);
    if (prediction < 0)
       num_wrong_face = num_wrong_face + 1; 
    end
end
for i = 1:size(nonfaces,3)
    prediction = boosted_predict(nonfaces(:, :, i), boosted_classifier, weak_classifiers, boosted_classifier_num);
    if (prediction > 0)
     num_wrong_nonface = num_wrong_nonface + 1;
    end
end

% Each time we create a weak classifer, we use generate_classifer

% Every weak classifer has soft classifer per threshold

% Top 1000 responses will become the  classifer

% get integral skin image


% generate classifer