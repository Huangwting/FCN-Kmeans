# ======1.建立工程并导入sklearn包=======
import numpy as np
import PIL.Image as image  # 加载PIL包，用于加载创建图片
from sklearn.cluster import KMeans  # 加载Kmeans算法

read_unit = 256.0
# =======2.加载图片并进行预处理========
def loadData(filePath):
    # 读取图像，并将图像中的pixel归一化到0-1
    f = open(filePath, 'rb')  # 以二进制形式打开文件
    data = []
    img = image.open(f)
    row, col = img.size  #图片大小
    for i in range(row): # 每个像素单位归一化0-1
        for j in range(col):
            x, y, z = img.getpixel((i, j))
            data.append([x / read_unit, y / read_unit, z / read_unit])  # 范围内并存入data
    f.close()
    imgData = np.mat(data)
    imgData = np.asarray(imgData)
    return imgData, row, col  # 以numpy形式返回imgData，以及data尺寸row, col


def save_new_pic(k):

    imgData, row, col = loadData('img1.jpg')  # 加载数据
    # 调节此参数
    n_clusters_num = k
    print("n_clusters_num " + str(n_clusters_num))

    # =======—3.加载Kmeans聚类算法========
    # 补全代码，使用 K-means方法进行分割预测
    mask = KMeans(n_clusters=n_clusters_num).fit_predict(imgData)

    # =======4.对像素点进行聚类并输出=======
    label = mask.reshape([row, col])
    pic_new = image.new("L", (row, col))  # 创建一张新的灰度图保存聚类后的结果
    for i in range(row):  # 根据所属类别向图中添加灰度值
        for j in range(col):
            pic_new.putpixel((i, j), int(256 / (label[i][j] + 1)))

    new_file_name = "img1_seg_" + str(n_clusters_num) + ".jpg"
    pic_new.save(new_file_name, "JPEG")  # 以JPEG格式保存图片
