restoredefaultpath;
clear all;
close all;

% Change path to your own local path
repo_path = 'C:\Users\Hawkk\OneDrive\college\CurrentClasses\computervision\FinalProject\ComputerVisionProj\Code\given';
training_faces = 'C:\Users\Hawkk\OneDrive\college\CurrentClasses\computervision\FinalProject\ComputerVisionProj\Data\training_faces'
main_path = 'C:\Users\Hawkk\OneDrive\college\CurrentClasses\computervision\FinalProject\ComputerVisionProj\Code\main'

s = filesep; % This gets the file separator character from the  system


addpath([repo_path s '00_common' s '00_detection'])
addpath([repo_path s '00_common' s '00_images'])
addpath([repo_path s '00_common' s '00_utilities'])
cd([repo_path])

training