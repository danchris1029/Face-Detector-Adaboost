function result = avi_read_frame(filename, frame)

% reads a double image frame from a video

vid = VideoReader(filename);
result = double(read(vid, frame));
