import numpy as np
from kmeans import AnchorKmeans
from datasets import AnnotParser
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
# 解析标签文件
# plt.style.use('ggplot')
annot_dir = "../../../../yolov4-tiny-pytorch/VOCdevkit/VOC2007/Annotations"
parser = AnnotParser('xml')
boxes = parser.parse_xml(annot_dir)
print('boxes shape : {}'.format(boxes.shape))

# k-means聚类
print('[INFO] Run anchor k-means with k = 2,3,...,10')
results = {}
for k in range(2, 11):
    model = AnchorKmeans(k, random_seed=333)
    model.fit(boxes)
    avg_iou = model.avg_iou()
    results[k] = {'anchors': model.anchors_, 'avg_iou': avg_iou}
    print("K = {}, Avg IOU = {:.4f}".format(k, avg_iou))

# 绘制曲线
print('[INFO] Plot average IOU curve')
plt.figure()
plt.plot(range(2, 11), [results[k]["avg_iou"] for k in range(2, 11)], "o-")
plt.ylabel("Avg IOU")
plt.xlabel("K")
plt.show()

# 打印出先验框形状
print('[INFO] The result anchors:')
best_k = 5
anchors = results[best_k]['anchors']
print(anchors)

print('[INFO] Visualizing anchors')
w_img, h_img = 608, 608

anchors[:, 0] *= w_img
anchors[:, 1] *= h_img
anchors = np.round(anchors).astype(np.int)

rects = np.empty((5, 4), dtype=np.int)
for i in range(len(anchors)):
    w, h = anchors[i]
    x1, y1 = -(w // 2), -(h // 2)
    rects[i] = [x1, y1, w, h]

# 可视化先验框形状
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot()
for rect in rects:
    x1, y1, w, h = rect
    rect1 = Rectangle((x1, y1), w, h, color='royalblue', fill=False, linewidth=2)
    ax.add_patch(rect1)
plt.xlim([-(w_img // 2), w_img // 2])
plt.ylim([-(h_img // 2), h_img // 2])

plt.show()