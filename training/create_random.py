from random import randrange
import numpy as np

# The `val_index.npy` must be created every time the number of training patches changes. It defines (and keeps set)
# which of the patches will be used for validation.

# This file must be changed if the DSen2_60 net is trained! (change the `path` and size of patches)

# Size: number of S2 tiles (times) patches per tile
size = 45*8000
ratio = .1
nb = int(size * ratio)

index = np.zeros(size).astype(np.bool)
i = 0
while np.sum(index.astype(np.int)) < nb:
    x = randrange(0, size)
    index[x] = True
    i += 1

path = '../data/train/'
np.save(path + 'val_index.npy', index)

print('Full no of samples: {}'.format(size))
print('Validation samples: {}'.format(np.sum(index.astype(np.int))))

print("Number of iterations: {}".format(i))
