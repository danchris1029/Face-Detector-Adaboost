restoredefaultpath;
clear all;
close all;

% Change path to your own local path
%Connor
project_path = 'C:\Users\Connor\Documents\MATLAB\ComputerVisionFinal\ComputerVisionProj';
repo_path = strcat(project_path,'\Code\given');
training_faces = strcat(project_path, '\Data\training_faces');
training_nonfaces = strcat(project_path, '\Data\training_nonfaces');
test_faces_photos = strcat(project_path, '\Data\test_face_photos');
test_cropped_faces = strcat(project_path, '\Data\test_cropped_faces');
test_nonfaces = strcat(project_path, '\Data\test_nonfaces');

main_path = strcat(project_path, '\Code\main');

s = filesep; % This gets the file separator character from the  system


addpath([repo_path s '00_common' s '00_detection'])
addpath([repo_path s '00_common' s '00_images'])
addpath([repo_path s '00_common' s '00_utilities'])
cd([repo_path])

train