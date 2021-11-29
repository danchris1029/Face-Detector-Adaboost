%% Create weak_classifers

function result = create_weak_classifers(weak_c, dimensions, number)

    temp_classifer = cell(1, number);
    
    for i = 1:number
        temp_classifer{i} = generate_classifier(dimensions(1), dimensions(2));
    end

    result = cat(2, weak_c, temp_classifer); 
    
end