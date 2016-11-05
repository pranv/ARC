import os

import numpy as np

from scipy.ndimage import imread


os.system('wget https://github.com/brendenlake/omniglot/archive/master.zip')
os.system('unzip -a master.zip')

path_bg = os.path.join('omniglot-master', 'python', 'images_background.zip')
path_ev = os.path.join('omniglot-master', 'python', 'images_evaluation.zip')

os.system('unzip -a ' + path_bg)
os.system('unzip -a ' + path_ev)


def omniglot_folder_to_NDarray(path_im):
    alphbts = os.listdir(path_im)
    ALL_IMGS = []

    for alphbt in alphbts:
        chars = os.listdir(os.path.join(path_im, alphbt))
        for char in chars:
            img_filenames = os.listdir(os.path.join(path_im, alphbt, char))
            char_imgs = []
            for img_fn in img_filenames:
                fn = os.path.join(path_im, alphbt, char, img_fn)
                I = imread(fn)
                I = np.invert(I)
                char_imgs.append(I)
            ALL_IMGS.append(char_imgs)

    return np.array(ALL_IMGS)


all_bg = omniglot_folder_to_NDarray('images_background')
all_ev = omniglot_folder_to_NDarray('images_evaluation')
all_imgs = np.concatenate([all_bg, all_ev], axis=0)

np.save('omniglot', all_imgs)

# clean up. just leave what is required.
os.system('rm master.zip')
os.system('rm -rf omniglot-master/')
os.system('rm -rf images_background/')
os.system('rm -rf images_evaluation/')

print 'omniglot data preperation done, exiting ...'
