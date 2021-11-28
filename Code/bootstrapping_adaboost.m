function boosted_classifier = bootstrapping_adaboost(dimensions,faces,nonfaces,face_pool,nonface_pool,number_faces,number_nonfaces)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
number = 1000;
weak_classifiers = cell(1, 0);

rounds = 20;
%for round = 1: rounds

old_num_wrong_nonface = -1;
old_num_wrong_face = -1;
num_wrong_face = 0;
num_wrong_nonface = 0;
delta_wrong_face = inf;
delta_wrong_nonface = inf;
boosted_classifier_num = 0;
while (delta_wrong_face > 20 && delta_wrong_nonface > 20)
    old_num_wrong_nonface = num_wrong_nonface;
    old_num_wrong_face = num_wrong_face;

    
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
    boosted_classifier_num = boosted_classifier_num + 5;
    boosted_classifier = AdaBoost(responses, labels, boosted_classifier_num);

    num_wrong_face = 0;
    for i = 1:size(face_pool,3)
        prediction = boosted_predict(face_pool(:, :, i), boosted_classifier, weak_classifiers, boosted_classifier_num);
        if (prediction < 0)
           num_wrong_face = num_wrong_face + 1; 
           faces = cat(3, face_pool(:, :, i), faces);
        end
    end
    
    num_wrong_nonface = 0;
    for i = 1:size(nonface_pool,3)
        prediction = boosted_predict(nonface_pool(:, :, i), boosted_classifier, weak_classifiers, boosted_classifier_num);
        if (prediction > 0)
           num_wrong_nonface = num_wrong_nonface + 1; 
           nonfaces = cat(3, nonface_pool(:, :, i), faces);
        end
    end
    
    % We need a non_face pool for false negatives
    kept_classifiers = cell(1, boosted_classifier_num);
    for i = 1:boosted_classifier_num
        kept_classifiers{i} = weak_classifiers{boosted_classifier(i,1)};
    end
    boosted_classifier = kept_classifiers;
    weak_classifiers = kept_classifiers;
    
    if(num_wrong_face == 0 && num_wrong_nonface == 0)
        break;
    end
    delta_wrong_face = abs(old_num_wrong_face - num_wrong_face);
    delta_wrong_nonface = abs(old_num_wrong_nonface - num_wrong_nonface);
end

end

