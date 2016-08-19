from  fli import processImg

import matplotlib.cm as cm
import matplotlib.pyplot as plt

img = processImg( '../data/pics/uic', '7.png' , save=False,
                  save_path="../data/transform/", flatten = False)

plt.title("Test",  fontsize=24)
plt.imshow(img, cmap = cm.binary)
plt.show()