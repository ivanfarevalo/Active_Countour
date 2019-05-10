#Active Contour

####Objective:
Implement a lip tracking program using the Snake or flexible template.

####Program Goals:
Be able to track the mouth motion with the following deformation and change:
- Talking
- Laughing, fawning and other common expressions
- Slight camera zoom in or out
- Slight rotation and translation of the head
- Slight change in the environmental illumination

####How to Run:
``python3 prog2.py -d directory_name -r root_name -idx1 first_index -idx2 second_index -t lip_template``

This program accepts four arguments: (directory, root, idx1, idx2, lip_template). “directory”
is the data directory, “root” is the root image filename, idx1 is the starting index, and idx2 is the
end index. Images in the video are named root_idx1.jpg to root_idx2.jpg, increment by 1.
The image index will always be 5 digits long, with 0 padding if necessary, e.g., 00001, 00010,
00100, etc. lip_template is the file that stores initial lip template. For each
input frame, the program will output the corresponding output frame with the lip tracking result highlighted in color. 

#### Output:
Folder output_images in project