import os
import sys
import numpy as np
from skimage import io


def listdir_fullpath(d):
    return [ os.path.join(d, f) for f in os.listdir(d) ]

def scale(M):
    M -= np.min(M)
    M /= np.max(M)
    M = (M * 255).astype(np.uint8)
    return M

filenames = listdir_fullpath(sys.argv[1])
imgs = np.concatenate([ io.imread(filename).reshape(1, -1) for filename in filenames ]).astype(np.float64)

'average face'
avg_img = np.mean(imgs, axis=0)

'Problem1'
'''
avg_img = avg_img.reshape(600, 600, 3).astype(np.uint8)
io.imsave('../result/avg_face.jpg', avg_img)
'''

'PCA'
norm_imgs = imgs - avg_img
U, s, V = np.linalg.svd(norm_imgs, full_matrices=False)

'Problem4'
'''
sum_s = s.sum()
r1 = np.around(s[0]/sum_s, decimals=3)
r2 = np.around(s[1]/sum_s, decimals=3)
r3 = np.around(s[2]/sum_s, decimals=3)
r4 = np.around(s[3]/sum_s, decimals=3)
print(r1, r2, r3, r4)
'''


'Problem2'
'''
for i in range(4):
    eigen_img = -1 * V[i]
    eigen_img = scale(eigen_img)
    eigen_img = eigen_img.reshape(600, 600, 3)
    io.imsave('../result/eigen_face%d.jpg' % i, eigen_img)
'''

'reconstruct'
'Problem3'
img = io.imread(os.path.join(sys.argv[1], sys.argv[2])).reshape(1, -1).astype(np.float64)
img = img - avg_img
w = np.dot(img, V[:4].T).flatten()
recon_img = np.sum([w[i] * V[i] for i in range(len(w))], axis=0, dtype=np.float64)
recon_img = recon_img + avg_img
recon_img =  scale(recon_img)
recon_img = recon_img.reshape(600, 600, 3)
io.imsave('./reconstruction.jpg', recon_img)
