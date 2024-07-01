import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from pydicom import FileDataset
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
from plotly.tools import FigureFactory as FF
from plotly.graph_objs import *
from PIL import Image
import imageio
import threading
import time
from concurrent.futures import ThreadPoolExecutor


'''
Hàm load_scan(path) được thiết kế để tải các tệp DICOM từ một thư mục cụ thể và trả về một danh sách 
các đối tượng DICOM. Dưới đây là các bước chính trong hàm:

- Tạo một danh sách slices bằng cách đọc tất cả các tệp DICOM từ thư mục được chỉ định bởi path.
- Sắp xếp các phần tử trong danh sách slices theo số thứ tự của hình ảnh DICOM, 
được xác định bằng thuộc tính InstanceNumber.
- Tính toán độ dày của mỗi lát cắt dựa trên thông tin về vị trí hình ảnh DICOM 
(hoặc vị trí lát cắt, nếu thông tin về vị trí hình ảnh không có sẵn).
- Gán giá trị độ dày của lát cắt cho mỗi đối tượng DICOM trong danh sách.
Trả về danh sách đã cập nhật.
'''
def load_scan(path):
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

'''

Hàm get_pixels_hu(scans) được thiết kế để chuyển đổi dữ liệu pixel của hình ảnh DICOM 
thành đơn vị Hounsfield (HU), một đơn vị đo được sử dụng trong hình ảnh y tế để 
biểu thị độ mật độ của một vật chất trên hình ảnh y tế.

- Tạo một ma trận 3D (image) từ dữ liệu pixel của các hình ảnh trong scans.
- Chuyển đổi kiểu dữ liệu của image thành int16 để chuẩn hóa các giá trị pixel.
- Đặt các pixel nằm ngoài vùng quét (nếu có) thành giá trị 0.
- Chuyển đổi các giá trị pixel thành đơn vị Hounsfield (HU) bằng cách sử dụng các thông số RescaleIntercept và RescaleSlope của hình ảnh DICOM.
- Trả về mảng image đã được chuyển đổi thành HU.
'''
def get_pixels_hu(scans):
    image = np.stack([s.pixel_array for s in scans])
    image = image.astype(np.int16)
    image[image == -2000] = 0

    # Chuyển đổi thành HU
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def sample_stack(stack, rows=5, cols=5, start_with=10, show_every=3):
    fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
    
    def plot_slice(i):
        ind = start_with + i * show_every
        ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
        ax[int(i / rows), int(i % rows)].imshow(stack[ind], cmap='gray')
        ax[int(i / rows), int(i % rows)].axis('off')

    with ThreadPoolExecutor() as executor:
        _ = list(executor.map(plot_slice, range(rows * cols)))

    plt.show()


def resample(image, scan, new_spacing=[1, 1, 1]):
    x = scan[0].SliceThickness
    y = scan[0].PixelSpacing[0]
    z = scan[0].PixelSpacing[1]

    total_spacing = [x, y, z]

    spacing = map(float, total_spacing)
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor)

    return image, new_spacing

def make_lungmask(img, display=False):
    row_size = img.shape[0]
    col_size = img.shape[1]

    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std

    middle = img[int(col_size / 5):int(col_size / 5 * 4), int(row_size / 5):int(row_size / 5 * 4)]
    mean = np.mean(middle)
    max_val = np.max(img)
    min_val = np.min(img)

    img[img == max_val] = mean
    img[img == min_val] = mean

    # def kmeans_clustering(data):
    #     kmeans = KMeans(n_clusters=2, n_init=11).fit(data)
    #     centers = sorted(kmeans.cluster_centers_.flatten())
    #     threshold = np.mean(centers)
    #     return threshold

    def kmeans_clustering(data):
        # Thực hiện phân cụm K-means
        kmeans = KMeans(n_clusters=2, n_init=11).fit(data)
        centers = sorted(kmeans.cluster_centers_.flatten())
        threshold = np.mean(centers)
        return threshold

    def kmeans_clustering_multithreading(data):
        num_threads = 4
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Thực hiện phân cụm K-means trên từng phần của dữ liệu
            results = list(executor.map(kmeans_clustering, [data]))  # Truyền data như một danh sách
        # Tính toán ngưỡng cuối cùng từ các kết quả
        final_threshold = np.mean(results)
        return final_threshold
    
    def threshold_image(data, threshold):
        return np.where(data < threshold, 1.0, 0.0)

    def process_label(N, mask, labels):
        mask = mask + np.where(labels == N, 1, 0)
        return mask

    threshold = kmeans_clustering_multithreading(np.reshape(middle, [np.prod(middle.shape), 1]))
    thresh_img = threshold_image(img, threshold)

    eroded = morphology.erosion(thresh_img, np.ones([3, 3]))
    dilation = morphology.dilation(eroded, np.ones([8, 8]))

    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < row_size / 10 * 9 and B[3] - B[1] < col_size / 10 * 9 and B[0] > row_size / 5 and B[2] < col_size / 5 * 4:
            good_labels.append(prop.label)

    mask = np.ndarray([row_size, col_size], dtype=np.int8)
    mask[:] = 0

    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    mask = morphology.dilation(mask, np.ones([10, 10]))  

    if display:
        fig, ax = plt.subplots(3, 2, figsize=[12, 12])
        ax[0, 0].set_title("Original")
        ax[0, 0].imshow(img, cmap='gray')
        ax[0, 0].axis('off')
        ax[0, 1].set_title("Threshold")
        ax[0, 1].imshow(thresh_img, cmap='gray')
        ax[0, 1].axis('off')
        ax[1, 0].set_title("After Erosion and Dilation")
        ax[1, 0].imshow(dilation, cmap='gray')
        ax[1, 0].axis('off')
        ax[1, 1].set_title("Color Labels")
        ax[1, 1].imshow(labels)
        ax[1, 1].axis('off')
        ax[2, 0].set_title("Final Mask")
        ax[2, 0].imshow(mask, cmap='gray')
        ax[2, 0].axis('off')
        ax[2, 1].set_title("Apply Mask on Original")
        ax[2, 1].imshow(mask * img, cmap='gray')
        ax[2, 1].axis('off')
        plt.show()

    return mask * img, mask

data_path = "D:\hien_code\dicom_image\dicom_data"

#working_path: đường dẫn đến folder làm việc, để lưu mọi thứ
output_path = "D:/hien_code/project_tinh_toan_song_song/output_folder_para/"
total_CT_image = glob(data_path + '/*.dcm')

# Check thử đường dẫn
# print ("Total of %d DICOM images.\nFirst 5 filenames:" % len(g))
# print ('\n'.join(g[:5]))


# '''THỬ VỚI FILE CT000019'''
# id = 25
# # Chạy qua 2 hàm tiền xử lý ảnh
# start_time = time.time()
# os.environ['LOKY_MAX_CPU_COUNT'] = '3'
# patient = load_scan(data_path)
# imgs = get_pixels_hu(patient)

# file_name = os.path.basename(total_CT_image[id])
# desired_part = file_name.split('.')[0]
# # print(desired_part)  # Kết quả: 'CT000019'
# # Taọ thư mục có tên giống tên file DICOM
# new_folder_path_DICOM_name = output_path + desired_part

# try:
#     os.makedirs(new_folder_path_DICOM_name, exist_ok=True)
# except FileExistsError:
#     print("Directory already exists:", new_folder_path_DICOM_name)

# # Thêm / cho working_path xD

# working_path = new_folder_path_DICOM_name +'/'

# # lưu file đã qua xử lý vào working_path
# np.save(os.path.join(working_path, "fullimages_%d.npy" % id), imgs)

# file_used=working_path + "fullimages_%d.npy" % id
# imgs_to_process = np.load(file_used).astype(np.float64)

# #in thử ra kiểm tra
# imgs_to_process = np.load(working_path+'fullimages_{}.npy'.format(id))

# sample_stack(imgs_to_process, 5, 5, 20, 3)

# imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1,1,1])

# #lưu file
# masked_lung = []
# for img in imgs_after_resamp:
#     lung, mask = make_lungmask(img)
#     # lung, mask = result_queue.get()  # Lấy kết quả từ hàng đợi
#     masked_lung.append(lung)

# for i in range(0, len(masked_lung), 1):
#     masked_image = (masked_lung[i] * 255).astype(np.uint8)
#     output_img = working_path + "maskedimage{}.jpg".format(i)
#     imageio.imwrite(output_img, masked_image)

# end_time=time.time()
# total_time = end_time - start_time
# print("Tổng thời gian chương trình chạy:", total_time, "giây")






'''THỨ VỚI 5 FILE ĐẦU TIÊN'''
start_time = time.time()
for i in range(5):
    '''tạo 1 folder giống tên file DICOM mình đang xử lý
    để lưu file numpy, và lưu ảnh xong khi xử lý
    '''
    # Chạy qua 2 hàm tiền xử lý ảnh
    os.environ['LOKY_MAX_CPU_COUNT'] = '4'
    patient = load_scan(data_path)
    imgs = get_pixels_hu(patient)

    file_name = os.path.basename(total_CT_image[i])
    desired_part = file_name.split('.')[0]
    # print(desired_part)  # Kết quả: 'CT000019'
    # Taọ thư mục có tên giống tên file DICOM
    new_folder_path_DICOM_name = output_path + desired_part

    try:
        os.makedirs(new_folder_path_DICOM_name, exist_ok=True)
    except FileExistsError:
        print("Directory already exists:", new_folder_path_DICOM_name)

    # Thêm / cho working_path xD
    working_path = new_folder_path_DICOM_name +'/'

    # lưu file đã qua xử lý vào working_path
    np.save(os.path.join(working_path, "fullimages_%d.npy" % i), imgs)

    file_used=working_path + "fullimages_%d.npy" % i
    imgs_to_process = np.load(file_used).astype(np.float64)

    #in thử ra kiểm tra
    imgs_to_process = np.load(working_path+'fullimages_{}.npy'.format(i))

    sample_stack(imgs_to_process, 5, 5, 20, 3)

    imgs_after_resamp, spacing = resample(imgs_to_process, patient, [1,1,1])

    #lưu file
    masked_lung = []
    just_mask = []

    for img in imgs_after_resamp:
        lung, mask = make_lungmask(img)
        masked_lung.append(lung)
        just_mask.append(mask)

    for i in range(0, len(masked_lung), 1):
        masked_image = (masked_lung[i] * 255).astype(np.uint8)
        output_img = working_path + "maskedimage{}.jpg".format(i)
        imageio.imwrite(output_img, masked_image)
        
    for i in range(0, len(just_mask), 1):
        just_mask_image = (just_mask[i] * 255).astype(np.uint8)
        output_img = working_path + "justmaskedimage{}.jpg".format(i)
        imageio.imwrite(output_img, just_mask_image)
end_time=time.time()
total_time = end_time - start_time
print("Tổng thời gian chương trình chạy:", total_time, "giây")