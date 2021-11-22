function result = avi_read_gray(filename, frame)

% reads a double image frame from a video

vid = VideoReader(filename);
result = double_gray(read(vid, frame));