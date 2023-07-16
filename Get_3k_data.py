# 2k train data, 1k test data
def make_data(subset_name, start_idx, end_idx):
    for category in ('cat', 'dog'):
        new_dir = new_base_dir / subset_name / category
        org_dir = original_dir / category
        image_names = [f'{i}.jpg' for i in range(start_idx, end_idx)]
        for name in image_names:
            shutil.copyfile(src=org_dir / name, dst=new_dir / name)

import os, pathlib, shutil, sys
import numpy as np

start_idx = np.random.randint(0, 8000)  # train data
end_idx = start_idx + 1000

start_idx2 = end_idx   # test data
end_idx2 = end_idx + 500

original_dir = pathlib.Path.cwd().parent / 'PetImages'
new_base_dir = pathlib.Path.cwd().parent / 'new_cats_dogs'

if __name__ == '__main__':
    make_data('train', start_idx, end_idx)
    make_data('test', start_idx2, end_idx2)
