function [result, labels] = create_responses(weak_class, faces, nonfaces, face_pool, nonface_pool, ...
                                    number_faces,number_nonfaces, dimensions)

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

    classifier_number = numel(weak_class);

    responses = zeros(classifier_number, example_number);

    for i = 1:example_number
        %integral = examples(:, :, example);
        integral = examples(:, :, i);
        for feature = 1:classifier_number
            classifier = weak_class {feature};
            responses(feature, i) = eval_weak_classifier(classifier, integral);
        end
        %disp(i)
    end
    
    result = responses;
    
end