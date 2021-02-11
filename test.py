import cv2
import numpy as np

w = 5
h = 5
delta = 3

cap = cv2.VideoCapture("/home/ghaskell/projects_Git/cuDDM/data/colloid_0.5um_vid.mp4")

frame1 = np.average(cap.read()[1], axis=2, weights=[1, 0, 0])[:w, :h]

while (delta >= 0):
    frame2 = np.average(cap.read()[1], axis=2, weights=[1, 0, 0])[:w, :h]
    delta -= 1

# for x in range(w):
#     for y in range(h):
#         frame1[x,y] = x
#         frame2[x,y] = y

diff_local = abs(frame1 - frame2)
norm_factor = 1 / (w*h)
print(norm_factor)


fft_temp_diff = abs(np.fft.fft2(diff_local))

print(diff_local.shape)

print(diff_local)

for x in range(w):
    for y in range(h):
        print(f"{frame1[x,y]}, {frame2[x,y]}, {diff_local[x,y]}, {fft_temp_diff[x,y]}")
