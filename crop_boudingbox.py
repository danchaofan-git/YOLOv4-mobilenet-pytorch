# - *-coding=utf-8 -*-
'''
将瑕疵图片的boundingbox截取下来作为新的一张图片
同时resize到跟原始图片统一大小
'''

import cv2
import os

def show_pic(img, bboxes=None):
    '''
    输入:
        img:图像array
        bboxes:图像的所有boudning box list, 格式为[[x_min, y_min, x_max, y_max]....]
        names:每个box对应的名称
    '''
    cv2.imwrite('./1.jpg', img)
    img = cv2.imread('./1.jpg')
    for i in range(len(bboxes)):
        bbox = bboxes[i]
        x_min = bbox[0]
        y_min = bbox[1]
        x_max = bbox[2]
        y_max = bbox[3]
        cv2.rectangle(img,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),3) 
    cv2.namedWindow('pic', 0)  # 1表示原图
    cv2.moveWindow('pic', 0, 0)
    cv2.resizeWindow('pic', 1200,800)  # 可视化的图片大小
    cv2.imshow('pic', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 
    os.remove('./1.jpg')

def crop_bd(img, bboxes):
        '''
        裁剪后的图片为boudingbox围起来的区域
        输入:
            img:图像array
            bboxes:该图像包含的所有boundingboxs,一个list,每个元素为[x_min, y_min, x_max, y_max],要确保是数值
        输出:
            crop_bds:裁剪后的boudingbox图像array list,（可能有多个boudingbox）
        '''
        #---------------------- 裁剪图像 ----------------------
        bud_bds = list()
        flower_bds = list()
        for bbox in bboxes:
            x_min = bbox[0]
            y_min = bbox[1]
            x_max = bbox[2]
            y_max = bbox[3]
            if bbox[4] == "after":
                crop_bd = img[y_min:y_max, x_min:x_max]
                flower_bds.append(crop_bd)
            elif bbox[4] == "first":
                crop_bd = img[y_min:y_max, x_min:x_max]
                bud_bds.append(crop_bd)
        return bud_bds,flower_bds

if __name__ == '__main__':

    from xml_helper import *
    import shutil
    # 存放原始图片和对应xml文件的目录
    source_bad_pic_root_path = './VOCdevkit/VOC2007/JPEGImages'
    source_xml_root_path = './VOCdevkit/VOC2007/Annotations'
    # 这是处理后存放的路径
    target_bud_pic_root_path = './VOCdevkit/VOC2007/CropImages/bud'      # 切割后的花苞图片
    if os.path.exists(target_bud_pic_root_path):
        shutil.rmtree(target_bud_pic_root_path)
    os.mkdir(target_bud_pic_root_path)

    target_flower_pic_root_path = './VOCdevkit/VOC2007/CropImages/flower'  # 切割后的花朵的图片
    if os.path.exists(target_flower_pic_root_path):
        shutil.rmtree(target_flower_pic_root_path)
    os.mkdir(target_flower_pic_root_path)

    cnt = 0
    process_flag = 0
    for parent, _, files in os.walk(source_bad_pic_root_path):
        for file in files:
            process_flag += 1
            print(str(process_flag)+'/'+str(len(files)))
            bad_pic_path = os.path.join(parent, file)
            xml_path = os.path.join(source_xml_root_path, file[:-4]+'.xml')
            img = cv2.imread(bad_pic_path)
            bboxes,shape = parse_xml(xml_path)

            # #原图可视化一下
            # show_pic(img,bboxes)

            bud_imgs,flower_imgs = crop_bd(img, bboxes)
            for croped_img in bud_imgs:
                cnt += 1
                target_pic_path = os.path.join(target_bud_pic_root_path, file[:-4]+'_croped_bud'+str(cnt)+'.jpg')
                # #reize一下
                # croped_img = cv2.resize(croped_img, (img.shape[1], img.shape[0]))
                # #可视化一下截取的图
                # show_pic(croped_img,[[0,0,croped_img.shape[1],img.shape[0]]])
                #写入
                cv2.imwrite(target_pic_path, croped_img)
                img = cv2.imread(target_pic_path)
                size = img.shape
                if size[0] < 224 or size[1] < 224:
                    os.remove(target_pic_path)

            for croped_img in flower_imgs:
                cnt += 1
                target_pic_path = os.path.join(target_flower_pic_root_path, file[:-4]+'_croped_flower'+str(cnt)+'.jpg')
                # #reize一下
                # croped_img = cv2.resize(croped_img, (img.shape[1], img.shape[0]))
                # #可视化一下截取的图
                # show_pic(croped_img,[[0,0,croped_img.shape[1],img.shape[0]]])
                #写入
                cv2.imwrite(target_pic_path, croped_img)
                img = cv2.imread(target_pic_path)
                size = img.shape
                # 去除那些尺寸小于224x224的
                if size[0] < 224 or size[1] < 224:
                    os.remove(target_pic_path)
                
