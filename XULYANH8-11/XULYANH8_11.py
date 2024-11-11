import cv2
import numpy as np
from skimage import io, filters
from matplotlib import pyplot as plt
import os

# Đường dẫn tới ảnh
duong_dan_anh = r'C:\Users\HUY\source\repos\XULYANH8-11\anhvetinh.png'

# Tải ảnh ở chế độ thang độ xám
anh = cv2.imread(duong_dan_anh, cv2.IMREAD_GRAYSCALE)
if anh is None:
    print("Lỗi: Không tìm thấy ảnh tại đường dẫn đã chỉ định.")
    exit()

# Áp dụng bộ lọc Gaussian để giảm nhiễu trước khi phát hiện biên
bo_loc_gaussian = cv2.GaussianBlur(anh, (5, 5), 0)

# Phát hiện biên Sobel
sobelx = cv2.Sobel(bo_loc_gaussian, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(bo_loc_gaussian, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)

# Phát hiện biên Prewitt
prewittx = filters.prewitt_h(bo_loc_gaussian)
prewitty = filters.prewitt_v(bo_loc_gaussian)
prewitt = np.hypot(prewittx, prewitty)

# Phát hiện biên Roberts
roberts = filters.roberts(bo_loc_gaussian)

# Phát hiện biên Canny
canny = cv2.Canny(bo_loc_gaussian, 100, 200)

# Hiển thị kết quả
plt.figure(figsize=(12, 8))

# Ảnh gốc
plt.subplot(231), plt.imshow(anh, cmap='gray')
plt.title('Ảnh gốc'), plt.axis('off')

# Kết quả Sobel
plt.subplot(232), plt.imshow(sobel, cmap='gray')
plt.title('Phát hiện biên Sobel'), plt.axis('off')

# Kết quả Prewitt
plt.subplot(233), plt.imshow(prewitt, cmap='gray')
plt.title('Phát hiện biên Prewitt'), plt.axis('off')

# Kết quả Roberts
plt.subplot(234), plt.imshow(roberts, cmap='gray')
plt.title('Phát hiện biên Roberts'), plt.axis('off')

# Kết quả Canny
plt.subplot(235), plt.imshow(canny, cmap='gray')
plt.title('Phát hiện biên Canny'), plt.axis('off')

plt.show()
