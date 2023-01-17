import pickle
import os
import cv2

# with open("./data/sunrgbd/sunrgbd_train_test_data/5050.pkl", 'rb') as f:
#     data = pickle.load(f)
# print('error')

dir1 = '/mnt1/myeongah/Test/Implicit3DUnderstanding-main_v1_det/out/total3d/20110611514267/output_imgs_det'
dir2 = '/mnt1/myeongah/Test/Implicit3DUnderstanding-main/out/total3d/20110611514267/output_imgs'
dir3 = '/mnt1/myeongah/Test/Implicit3DUnderstanding-main_v1_det/out/total3d/20110611514267/output_imgs_det_gt'

dir = '/mnt1/myeongah/Test/Implicit3DUnderstanding-main_v1_det/out/total3d/20110611514267/output_imgs_compare'
for i in range(5051):
    if os.path.isfile(os.path.join(dir2, '%d_bbox.png'%(i))) and os.path.isfile(os.path.join(dir1, '%d_bbox.png'%(i))):
        im1 = cv2.imread(os.path.join(dir1, '%d_bbox.png'%(i)))
        im2 = cv2.imread(os.path.join(dir2, '%d_bbox.png'%(i)))
        im3 = cv2.imread(os.path.join(dir3, '%d_bbox.png'%(i)))

        cv2.imwrite(os.path.join(dir, '%d_bbox.png'%(i)), cv2.hconcat([im1, im2, im3]))