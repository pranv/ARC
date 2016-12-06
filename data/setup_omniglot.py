import os

import numpy as np

from scipy.ndimage import imread


os.system('wget https://github.com/brendenlake/omniglot/archive/master.zip')
os.system('unzip -a master.zip')

path_bg = os.path.join('omniglot-master', 'python', 'images_background.zip')
path_ev = os.path.join('omniglot-master', 'python', 'images_evaluation.zip')
path_os = os.path.join('omniglot-master', 'python', 'one-shot-classification', 'all_runs.zip')

os.system('unzip -a ' + path_bg)
os.system('unzip -a ' + path_ev)
os.system('unzip -a ' + path_os)


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

TRIALS = []
LABELS = []
for run_num in range(1, 21):
    path = str(run_num)
    if len(path) == 1:
        path = '0' + path
    path = 'run' + path
    support_set = np.zeros((20, 105, 105), dtype='uint8')
    for img_num in range(1, 21):
        name = str(img_num)
        if len(name) == 1:
            name = '0' + name
        name = 'training/' + 'class' + name + '.png'
        filename = os.path.join(path, name)
        I = imread(filename)
        I = np.invert(I)
        support_set[img_num - 1] = I
        
    test_set = np.zeros((20, 105, 105), dtype='uint8')
    for img_num in range(1, 21):
        name = str(img_num)
        if len(name) == 1:
            name = '0' + name
        name = 'test/' + 'item' + name + '.png'
        filename = os.path.join(path, name)
        I = imread(filename)
        I = np.invert(I)
        test_set[img_num - 1] = I
    
    key_f = open(path + '/class_labels.txt', 'r')
    keys = key_f.readlines()
    key_f.close()
    matches = [int(key[-7:-5]) - 1 for key in keys]

    run = np.zeros((40 * 20, 105, 105), dtype='uint8')
    for i in range(20):
        k = i * 20
        run[k:k+20] = support_set
        k = 400 + i * 20
        run[k:k+20] = test_set[i]
    TRIALS.append(run)
    LABELS.append(matches)

    os.system('rm -rf ' + path)

X_OS = np.array(TRIALS)
y_OS = np.array(LABELS, dtype='int32')

os.mkdir('one_shot')
np.save('one_shot/X', X_OS)
np.save('one_shot/y', y_OS)

# clean up. just leave what is required.
os.system('rm master.zip')
os.system('rm -rf omniglot-master/')
os.system('rm -rf images_background/')
os.system('rm -rf images_evaluation/')

print 'omniglot data preperation done, exiting ...'
