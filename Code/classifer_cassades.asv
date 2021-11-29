function classifer_cassades layers = (f, d, ftarget, p, n, dimensions,faces,nonfaces,face_pool,nonface_pool,number_faces,number_nonfaces)

F = zeros(100);
D = zeros(100);
F(0) = 1.0;
D(0) = 1.0;

i = 0;

while(F(i) > ftarget):
    i = i + 1; 
    n(i) = 0;
    f(i) = f(i - 1);
    
    while(f(i) > f * f(i - 1)):
        n(i) = n(i) + 1;
        % evaluate within this function until function is met
        boostedbootstrapping_adaboost();
        
    N = [];
    if(F(i) > ftarget):
        %evaluate the current cascaded detector on the set of non-face 
        %images and put any false detections into the set N.
        
        
        
        
    