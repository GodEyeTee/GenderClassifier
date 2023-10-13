import os
import numpy as np
import shutil

def copyImages(source_path, des_path, filenames):
     if not(os.path.isdir(des_path)):
          os.mkdir(des_path)
     for i in range(len(filenames)):
          source_filename = f"{source_path}/{filenames[i]}"
          des_filename = f"{des_path}/{filenames[i]}"
          print(f"\rcopy {source_filename} to {des_filename} ... {i+1}/{len(filenames)}", end=" "*10)
          shutil.copy(source_filename, des_filename)

if __name__ == '__main__':
     
     # set parameters
     data_path = "../data/New_InTheWildFaces"
     save_path = "../data/Split_New_InTheWildFaces"
     ratio = [0.70, 0.15, 0.15] # train/val/test ratio
     
     # make dir
     if not(os.path.isdir(save_path)):
          os.mkdir(save_path)
     if not(os.path.isdir(f"{save_path}/train")):
          os.mkdir(f"{save_path}/train")
     if not(os.path.isdir(f"{save_path}/val")):
          os.mkdir(f"{save_path}/val")
     if not(os.path.isdir(f"{save_path}/test")):
          os.mkdir(f"{save_path}/test")
     
     # list class
     classes = os.listdir(data_path)
     
     # process each class
     for ic in range(len(classes)):
          age_fold = os.listdir(f"{data_path}/{classes[ic]}")
          
          for ic_age in range(len(age_fold)):
               # list filenames
               filenames = np.array(os.listdir(f"{data_path}/{classes[ic]}/{age_fold[ic_age]}"))

               # shuffle
               filenames = filenames[np.random.permutation(len(filenames))]
               
               # split data
               n_all = len(filenames)
               n_train = int(np.floor(ratio[0]*n_all))
               n_val = int(np.floor(ratio[1]*n_all))
               n_test = n_all - n_train - n_val
               train_filenames = filenames[:n_train]
               val_filenames = filenames[n_train:(n_train+n_val)]
               test_filenames = filenames[(n_train+n_val):]
               
               if not(os.path.isdir(f"{save_path}/train/{classes[ic]}")):
                    os.mkdir(f"{save_path}/train/{classes[ic]}")
               if not(os.path.isdir(f"{save_path}/test/{classes[ic]}")):
                    os.mkdir(f"{save_path}/test/{classes[ic]}")
               if not(os.path.isdir(f"{save_path}/val/{classes[ic]}")):
                    os.mkdir(f"{save_path}/val/{classes[ic]}")
               
               # copy images
               copyImages(f"{data_path}/{classes[ic]}/{age_fold[ic_age]}", f"{save_path}/train/{classes[ic]}", train_filenames)
               copyImages(f"{data_path}/{classes[ic]}/{age_fold[ic_age]}", f"{save_path}/val/{classes[ic]}", val_filenames)
               copyImages(f"{data_path}/{classes[ic]}/{age_fold[ic_age]}", f"{save_path}/test/{classes[ic]}", test_filenames)