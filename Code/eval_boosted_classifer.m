function [detection_rate, false_positive_rate, faces, nonfaces] = eval_boosted_classifer(boosted_class, weak_class, faces, nonfaces, face_pool, nonface_pool, ...
                                    number_faces,number_nonfaces, dimensions, boosted_classifier_num)

    num_wrong_face = 0;
    num_wrong_nonface = 0;
    
    detections = 0;
    
    % find false negative part
    for i = 1:size(face_pool,3)
        prediction = boosted_predict(face_pool(:, :, i), boosted_class, weak_class, boosted_classifier_num);
        if (prediction < 0)
           num_wrong_face = num_wrong_face + 1; 
           
           % check this part
           %faces = cat(3, face_pool(:, :, i), faces);
           
        else
            detections = detections + 1;
        end
    end
    
    % find false positive part
    for i = 1:size(nonface_pool,3)
        prediction = boosted_predict(nonface_pool(:, :, i), boosted_class, weak_class, boosted_classifier_num);
        if (prediction > 0)
           num_wrong_nonface = num_wrong_nonface + 1; 
           nonfaces = cat(3, nonface_pool(:, :, i), nonfaces);
           
           detections = detections + 1;
        end
    end
    
    detection_rate = detections / (size(face_pool, 3) + size(nonface_pool, 3));
    false_positive_rate = num_wrong_nonface / size(nonface_pool, 3);
    
end
    
    