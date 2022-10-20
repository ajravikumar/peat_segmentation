from gettext import npgettext
import os
import shutil

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

root_dir = '/home/ajay/Documents/Copyof_data/'

# list_masks = os.path.join(root_dir, 'dataset_mask')

list_masks = os.listdir(os.path.join(root_dir, 'dataset_mask/dataset_256/dataset'))

del_img = 0 
 
x=[]

for mask in list_masks:
    img = Image.open(os.path.join(root_dir, 'dataset_mask/dataset_256/dataset', mask))

    img_num = np.array(img)

    # import pdb; pdb.set_trace()

    x.append(np.count_nonzero(img_num))

    if np.count_nonzero(img_num) < 200:
        os.remove(os.path.join(root_dir, 'dataset_mask/dataset_256/dataset', mask))
        os.remove(os.path.join(root_dir, 'dataset_tif/dataset_256/dataset', mask))

        del_img +=1

plt.hist(x, bins=100)  # density=False would make counts
plt.ylabel('num of images')
plt.xlabel('non-zero pixels')

plt.show()

print(del_img ) 






