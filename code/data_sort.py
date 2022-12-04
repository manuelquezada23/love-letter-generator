import os
import shutil

current_folder = os.getcwd() 

forms = []
for folder in os.listdir("/Users/juanpabloramos/Desktop/love-letter-generator/data/topics"):
    if folder != ".DS_Store":
        if folder != "data_sort.py":
            if folder != "dadatum":
                forms.append(folder)

content_list = {}
for index, val in enumerate(forms):
    path = os.path.join(current_folder, val)
    content_list[ forms[index] ] = os.listdir(path)

# loop through the list of folders
for sub_dir in content_list:
  
    # loop through the contents of the
    # list of folders
    for contents in content_list[sub_dir]:
  
        # make the path of the content to move 
        path_to_content = sub_dir + "/" + contents  
  
        # make the path with the current folder
        dir_to_move = os.path.join(current_folder, path_to_content )
  
        # move the file
        shutil.move(dir_to_move, "/Users/juanpabloramos/Desktop/love-letter-generator/data/topics/dadatum")