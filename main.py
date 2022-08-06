from bs4 import BeautifulSoup
import requests
import os
import re
from selenium import webdriver
import time
from selenium.webdriver.chrome.options import Options

# 这个爬虫用于我深度学习制作训练集
# 这个爬虫只适用于"https://cn.bing.com/images“
# 然后在搜索之后的网址放在下面的URL中，我这个是cat的图片
if (os.path.isdir('C://cat'))!=True:
    os.mkdir('C://cat')

driver=webdriver.Edge()
URL="https://cn.bing.com/images/search?q=cat&qs=n&form=QBIR&sp=-1&pq=cat&sc=10-3&cvid=75C8638884E84DA79F2E2C8E45E99319&ghsh=0&ghacc=0&first=1&tsc=ImageHoverTitle"
# head = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/103.0.5060.134 Safari/537.36 Edg/103.0.1264.77"}
i=0
driver.get(URL)
# 因为bing上图片随着滚轮才会刷新更多的内容，这个用selenium模拟滑轮下滑以刷新更多图片的信息，过程是会显示在屏幕上的，当然你可以修改sleep的值，但是可能会拉不到底，效果不好
while True:
    h_before = driver.execute_script('return document.body.scrollHeight;')
    driver.execute_script(f'window.scrollTo(0,{h_before})')
    time.sleep(0.5)
    h_after = driver.execute_script('return document.body.scrollHeight;')
    # i+=1
    if h_after==h_before:
        break

html=driver.page_source
# print(html)
# print("ok")

# url="https://tse2-mm.cn.bing.net/th/id/OIP-C.brm1sy5RPkX0PTDt9Kjr3wHaEo?w=299&h=186&c=7&r=0&o=5&dpr=1.5&pid=1.7"
# ur="http(.*)"
# print(re.search(ur, url))
# html=requests.get(URL,headers=head).text




# 下面下载默认是简略图

Img=re.compile(r'img.*src="(.*?)"')
soup=BeautifulSoup(html,"html.parser")
data=[]
sfa=soup.find_all("img",{"class":"mimg","src":re.compile("http(.*)")})
# end=soup.find_all("div",{"id":"mmComponent_images_1_exp"})
# print("\n\n")
# print(sfa)
print(len(sfa))
for item in sfa:
    item=str(item)
    Picture=re.findall(Img,item)
    for b in Picture:
        data.append(b)

print(len(data))
i=0
# 下面可以控制爬多少个，修改range（）里面参数即可，不过我是全爬，就无所谓了，这个网页一页到头应该会有700多张，不能太多
for i in range(len(data)):
    print(i,data[i])
    r = requests.get(data[i])
    with open('C://cat/image{}.png'.format(i), 'wb') as f:#下载位置
        f.write(r.content)



# 下面下载是高清图
# Img=re.compile(r'"murl":"(.*?)"')
# soup=BeautifulSoup(html,"html.parser")
# data=[]
# sfa=soup.find_all("a",{"class":"iusc"})
# end=soup.find_all("div",{"id":"mmComponent_images_1_exp"})
# print("\n\n")
# print(sfa)
# print(len(sfa))
# for item in sfa:
#     item=str(item)
#     Picture=re.findall(Img,item)
#     for b in Picture:
#         data.append(b)
# # https://pic2.zhimg.com/v2-c772f409a8c2c5603df84c841274d83d_r.jpg
# # https://tse3-mm.cn.bing.net/th/id/OIP-C.pqaYseKD7xuemtxL7Cw13gHaHa?w=175&h=180&c=7&r=0&o=5&dpr=1.5&pid=1.7
# # print(len(data))
# # print(data)
#
# # 下面下载会有问题，因为高清图太大，下载速度过慢的原因，超过5s就会出现图片获取失败，
# b=0
# for i in range(len(data)):
#     try:
#         r = requests.get(data[i],stream=True,timeout=5)
#         with open('C://cat/image{}.png'.format(i), 'wb') as f:
#             for chunk in r.iter_content(chunk_size=32):
#                 f.write(chunk)
#     except Exception as e:
#         time.sleep(1)
#         print("本张图片获取异常，跳过...")
#
#     else:
#         print("第{}张".format(b))
#         b+=1

