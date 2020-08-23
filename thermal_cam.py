import time, board, busio
import numpy as np
import adafruit_mlx90640
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy import ndimage
import cv2
import matplotlib as mpl
mpl.rcParams['toolbar'] = 'None'

# Load phan detect khuong mat
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Ket noi camera Thường
cap = cv2.VideoCapture(0)

# Kết nối camera Nhiệt
i2c = busio.I2C(board.SCL, board.SDA, frequency=100000)
mlx = adafruit_mlx90640.MLX90640(i2c)
mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_8_HZ  # REFRESH_16_HZ # set refresh rate
mlx_shape = (24, 32)

# Phóng to kích thước ma trận nhiệt
mlx_interp_val = 10  # interpolate # on each dimension
mlx_interp_shape = (mlx_shape[0] * mlx_interp_val, mlx_shape[1] * mlx_interp_val)  # new shape

# Vẽ giao diện
fig = plt.figure(figsize=(8, 6))  # start figure
fig.canvas.set_window_title('Hệ thống giám sát nhiệt độ ra/vào')
fig.canvas.toolbar_visible = False
ax = fig.add_subplot(1, 2, 1)  # add subplot
ax2 = fig.add_subplot(1, 2, 2)
fig.subplots_adjust(0.05, 0.05, 0.95, 0.95)  # get rid of unnecessary padding

# Hiển thị ảnh nhiệt
therm1 = ax.imshow(np.zeros(mlx_interp_shape), interpolation='none',
                   cmap=plt.cm.bwr, vmin=25, vmax=45)  # preemptive image

# Hiển thị ảnh thường
therm2 = ax2.imshow(np.zeros((240, 320)), interpolation='none',
                    cmap=plt.cm.bwr, vmin=25, vmax=45)  # preemptive image

fig.canvas.draw()  # draw figure to copy background
ax_background = fig.canvas.copy_from_bbox(ax.bbox)  # copy background
fig.show()  # show the figure before blitting


# Hàm vẽ lại ảnh trên plot
def plot_update(x,y,w,h):
    # Khôi phục nền để xoá ảnh cũ
    fig.canvas.restore_region(ax_background)

    # Đọc ảnh nhiệt từ camera
    mlx.getFrame(frame)

    # Lật ảnh và phóng ảnh nhiệt
    data_array = np.fliplr(np.reshape(frame, mlx_shape))  # reshape, flip data
    data_array = ndimage.zoom(data_array, mlx_interp_val)  # interpolate

    # Vẽ ảnh nhiệt lên plot
    vmin = round(np.min(data_array), 2)
    vmax = round(np.max(data_array), 2)
    therm1.set_array(data_array)
    therm1.set_clim(vmin=vmin, vmax=vmax)
    ax.draw_artist(therm1)  # draw new thermal image

    # Tính nhiệt face
    vface_temp = round(np.max(data_array[y:y+h,x:w+x]), 2)
    if vface_temp<38:
        ax.text(250, -100, 'Nhiệt độ bình thường = ' + str(vface_temp) + "      ",
                bbox={'facecolor': 'yellow', 'alpha': 1, 'pad': 20})
    else:
        ax.text(250, -100, 'Có biểu hiện sốt. Nhiệt độ = ' + str(vface_temp) + "      ",
                bbox={'facecolor': 'red', 'alpha': 1, 'pad': 20})
    fig.canvas.blit(ax.bbox)
    fig.canvas.flush_events()

    return

plt.ion()
frame = np.zeros(mlx_shape[0] * mlx_shape[1])  # 768 pts
count = 0
while True:
    ret, img = cap.read()
    if ret:
        count = count + 1
        img = cv2.resize(img, dsize=(320, 240))
        img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if count%5==0:
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            if len(faces)==1:
                for (x, y, w, h) in faces:
                    cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
                try:
                    plot_update(x,y,w,h)  # update plot
                except:
                    continue
                therm2.set_data(img2)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
            break

plt.show()
cap.release()
cv2.destroyAllWindows()

