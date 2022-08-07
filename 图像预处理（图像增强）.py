import os
import re
import shutil
from PIL import Image
import torchvision.transforms as transforms

import random
size=64





# 无用的内容
################################################
# dog_path=r"C:\train\dog"
# cat_path=r"C:\train\cat"
# for c in range(2):
#     if c ==0:
#         i=r"C:\train\dog"
#         i_1="dog"
#     else:
#         i = r"C:\train\cat"
#         i_1 = "cat"
#     file_list=os.listdir(i)
#     print(file_list)
#     for f in range(len(file_list)):
#         new_name=file_list[f].replace(file_list[f],"%s%d.png"%(i_1,f))
#         os.rename(os.path.join(i,file_list[f]),os.path.join(i,new_name))
#
# print("end")
#######################################################################

# 图像增强
###############################################
def DataEnhance(sourth_path,aim_dir):
    name=0
    file_list=os.listdir(sourth_path)
    if not os.path.exists(aim_dir):
        os.makedirs(aim_dir)
    for i in file_list:
        img=Image.open("%s\%s"%(sourth_path,i))

        name+=1
        transform1=transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.Resize(size)
        ])
        img1=transform1(img)
        img1.save("%s\%s%s.png"%(aim_dir,i[:3],name))


        name+=1
        transform2=transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.5,contrast=0.5,hue=0.5)
        ])
        img2 = transform2(img)
        img2.save("%s\%s%s.png"%(aim_dir,i[:3],name))

        name+=1
        transform3=transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomCrop(227,pad_if_needed=True),
            transforms.Resize(size)
        ])
        img3 = transform3(img)
        img3.save("%s\%s%s.png"%(aim_dir,i[:3],name))

        name+=1
        transform4=transforms.Compose([
            transforms.ToTensor(),
            transforms.ToPILImage(),
            transforms.RandomRotation(60),
            transforms.Resize(size),
        ])
        img4 = transform4(img)
        img4.save("%s\%s%s.png"%(aim_dir,i[:3],name))

# DataEnhance(r"C:\train\cat",r"C:\train\cat1")
####################################################

# 从训练集里面随机选择1000张作为验证集
#################################################
def makevalid(source_path,purpose_path):
    file_list=os.listdir(source_path)
    name=0
    print(len(file_list))
    file_list=random.sample(file_list,1000)
    print(len(file_list))
    for i in file_list:
        name+=1
        # with open(i,"rb") as f:
        #     f.save(purpose_path+"%s%d.png"%(i[:3],name))
        shutil.copy(source_path+"\\"+i,purpose_path+"\\%s%d.png"%(i[:3],name))

makevalid(r"C:\train\dog",r"C:\valid\dog")
#############################################################














