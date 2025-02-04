% train.m

%% Add important info from paths for functions
s = filesep; % This gets the file separator character from the  system
training_faces = strcat(training_directory, '\training_faces');
training_nonfaces = strcat(training_directory, '\training_nonfaces');
other_code = strcat(code_directory, '\given');
addpath([other_code s '00_common' s '00_detection'])
addpath([other_code s '00_common' s '00_images'])
addpath([other_code s '00_common' s '00_utilities'])
addpath(code_directory)
addpath(other_code)
addpath(training_faces)
addpath(training_nonfaces)

best_boosted_classifer = zeros(0, 3);

%% Read in all faces and non faces

% we use a 1:2 ratio for faces and nonface subwindows.
% and every nonface image has 5 subwindows.
number_faces = 50;
number_nonfaces = 20;

face_images = dir(fullfile(training_faces,'*.bmp'));
nonface_images = dir(fullfile(training_nonfaces,'*.jpg'));

faces = zeros(63, 57, number_faces);
for i = 1:number_faces
  filename = fullfile(training_faces,face_images(i).name);
  tempface = read_gray(filename);
  faces(:,:,i) = tempface(26:88, 22:78);
end

face_pool = zeros(63, 57, size(face_images, 1));
for i = 1:size(face_images, 1)
  filename = fullfile(training_faces,face_images(i).name);
  tempface = read_gray(filename);
  face_pool(:,:,i) = tempface(26:88, 22:78);
end

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

% traing_face dimension is 100 x 100
dimensions = [size(faces(:,:,1),1), size(faces(:,:,1),2)];%[100, 100];


tic;
%% training classifer cascades

number = 1000;
weak_classifiers = cell(1, 0);

num_wrong_face = 0;
num_wrong_nonface = 0;
boosted_classifier_num = 0;

% false_postive_rates
F = [1.0];

% detection_rates
D = [1.0];

% number of classifers per layer
n = [0];

% the current layer that we're on
i = 1;

% the maximum acceptable false positive rate per layer.
f = 0.5;

% the minimum acceptable detection rate per layer.
d = 0.5;

% target overall false positive rate.
ftarget = 0.01;

% threshold decreaser
threshold_decrease = -1000;

% stores all classifers from each layer
boosted_classes = cell(1, 0);

while(F(i) >= ftarget)
    threshold_decrease = -1000;
    i = i + 1; 
    D = [D, 0];
    n = [n, 1];
    
    % appending previous fpr to the array of F
    F = [F, (F(i - 1))];
    
    % if the current layer's fpr is more than (previous layer's fpr * f)
    while(F(i) >= f * F(i - 1))
        curr_f_target = f * F(i - 1);
        curr_f_target
        
        % use P and N to train a classifier with n(i) features using AdaBoost
        disp("training false_positive_rate = " + F(i)) 
        
        n(i) = n(i) + 1;
        disp("number of current features is = " + n(i)) 
        % Create weak classifers
		weak_classifiers = create_weak_classifers(weak_classifiers,dimensions, number);
		
		% Create responses
		[responses, labels] = create_responses(weak_classifiers, faces, nonfaces, face_pool, ...
										nonface_pool,number_faces,number_nonfaces, dimensions);
		
		boosted_classifier_num = n(i);
		boosted_classifier = AdaBoost(responses, labels, boosted_classifier_num);
		 
        classifer_index = 1;
        eval_round = 1;
        detection_rate = 0;
        
        [detection_rate, false_positive_rate, faces, nonfaces] = eval_boosted_classifer(boosted_classifier, weak_classifiers, faces, nonfaces, face_pool, nonface_pool, ...
										number_faces,number_nonfaces, dimensions, boosted_classifier_num);
        D(i) = detection_rate;
        F(i) = false_positive_rate;                            
                                    
        % the current cascaded classifier has a detection rate of at least d × D(i-1)
        while(D(i) <= (d * D(i - 1)))
           curr_d_target = (d * D(i - 1));
           curr_d_target
           eval_round = eval_round + 1;  
           
           % Evaluate current cascaded classifier on validation set to determine F(i) and D(i)
           [detection_rate, false_positive_rate, faces, nonfaces] = eval_boosted_classifer(boosted_classifier, weak_classifiers, faces, nonfaces, face_pool, nonface_pool, ...
										number_faces,number_nonfaces, dimensions, boosted_classifier_num);
           D(i) = detection_rate;
           F(i) = false_positive_rate;
           disp("evaluating dectection_rate = " + detection_rate)
           
           % decrease threshold for the ith classifier (i.e. how many weak classifiers need to accept for strong classifier to accept)
           % until the current cascaded classifier has a detection rate of at least d × D(i-1) (this also affects F(i))
           if(D(i) <= (d * D(i - 1)))
               boosted_classifier(classifer_index, 3) = boosted_classifier(classifer_index, 3) + threshold_decrease;
               classifer_index = classifer_index + 1;
               if(classifer_index == size(boosted_classifier, 1) + 1)
                   classifer_index = 1;
               end
               disp("threshold = " + boosted_classifier(classifer_index, 3));
               if(D(i) < 0.01)
                   threshold_decrease = 1000;
               end
                   
           end
           
        end
        
    end

    disp("created layer")
    boosted_classes = cat(2, boosted_classes, {boosted_classifier});
end
%% Save data for training
toc; 

save boosted_classifier boosted_classifier
save weak_classifiers weak_classifiers
save boosted_classifier_num boosted_classifier_num
save boosted_classes boosted_classes

%%