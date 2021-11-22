% We're just using training images

% we don't find the best rectangle filter, but the best classifer when
% boosting
image1 = read_gray(training_faces+"\2463d171.bmp");

image2 = read_gray(training_faces+"\2463d261.bmp");
image3 = read_gray(training_faces+"\2463d331.bmp");

faces = zeros(100, 100, 3);
faces(:, :, 1) = image1;
faces(:, :, 2) = image2;
faces(:, :, 3) = image3;
dimensions = [100, 100];
% traing_face dimension is 100 x 100

% We have 5 thresholds per pixel
% so 10000 * 5 = 50000 classifers

number = dimensions(1) * dimensions(2);
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

example_number = size(faces, 3);
classifier_number = numel(weak_classifiers);

responses =  zeros(classifier_number, example_number);

for example = 1:example_number
    %integral = examples(:, :, example);
    integral = integral_image(faces(:, :, example));
    for feature = 1:classifier_number
        classifier = weak_classifiers {feature};
        responses(feature, example) = eval_weak_classifier(classifier, integral);
    end
    disp(example)
end

% sum the responses from each image for each classifier together into one
% row within responses

labels = [1, -1, 1];
boosted_classifier = AdaBoost(responses, labels', 15);







% Each time we create a weak classifer, we use generate_classifer

% Every weak classifer has soft classifer per threshold

% Top 1000 responses will become the  classifer

% get integral skin image


% generate classifer