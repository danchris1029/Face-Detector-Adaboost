restoredefaultpath;
clear all;
close all;

% Change path to your own local path
repo_path = 'C:\Users\Hawkk\OneDrive\college\CurrentClasses\computervision\FinalProject\ComputerVisionProj\Code\main';
training_faces = 'C:\Users\Hawkk\OneDrive\college\CurrentClasses\computervision\FinalProject\ComputerVisionProj\Data\training_faces'
s = filesep; % This gets the file separator character from the  system

addpath([repo_path s 'Code' s '00_common' s '00_detection'])
addpath([repo_path s 'Code' s '00_common' s '00_images'])
addpath([repo_path s 'Code' s '00_common' s '00_utilities'])
addpath([repo_path s 'Code' s '17_boosting'])
addpath([repo_path s 'Data' s '00_common_data' s 'frgc2_b'])
cd([repo_path s 'Data' s '00_common_data'])


training