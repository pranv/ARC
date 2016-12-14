import numpy as np

import os

import PIL
from PIL import Image


def image2pixelarray(filepath):
    F = PIL.Image.open(filepath).resize((64, 64), resample=PIL.Image.LANCZOS).convert('L')
    F = list(F.getdata())
    F = np.array(F)
    F = F.reshape((64, 64))
    return F


os.system('wget http://vis-www.cs.umass.edu/lfw/lfw.tgz')
os.system('tar -xvf lfw.tgz')

path = 'lfw'

FACES = []
COUNTS = []

people = os.listdir(path)
for person in people:
    person_path = os.path.join(path, person)
    faces = os.listdir(person_path)
    for face in faces:
        face_path = os.path.join(person_path, face)
        FACES.append(image2pixelarray(face_path))
    COUNTS.append(len(faces))

FACES = np.array(FACES)
COUNTS = np.array(COUNTS)

os.system('rm -rf lfw')
os.system('rm lfw.tgz')

os.mkdir('LFW')
np.save('LFW/faces', FACES)
np.save('LFW/counts', COUNTS)
