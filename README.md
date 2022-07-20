# YOLOV7
![image](https://user-images.githubusercontent.com/46515944/179973455-f817fd76-410b-4566-9485-461c26962805.png)

前言
====
💣本篇主要介紹如何訓練 Yolov7 ，有寫得不完整或者過於簡易省略步驟的話，再麻煩有問題的讀者們在底下留言啦💣

> 前半部會先小小補充 Yolov7的介紹，若沒有興趣閱讀者，可直接跳至💎💎💎💎橫線區，感謝~

⌚我的顯卡為 NVDIA GeForce 2080，batch = 4，epoch = 300，耗時約 900 分鐘，2100張圖片訓練，3個類別，準確度 76%，給各位參考一下。由於訓練集屬於病患資料，極具隱私的考量，所以不進一步做分享。

⌨版本一覽：CUDA = 11.2；Python = 3.9；torchvision = 0.8.2+cu110；torch = 1.7.1+cu110

❌另外，本文提供的程式碼含有參數調整，所以你要記得改掉!!!此外，橫槓(—)在這邊出現是錯的，你要改成兩個小橫槓(--)，不然你直接複製去執行會出錯，我有說哦，沒看到的話，就....自己debug囉

📒小補充一下，這次的作者你們知道是誰嗎?!!是Yolov4原本的團隊啦
Chien-Yao Wang, Alexey Bochkovskiy, Hong-Yuan Mark Liao。
其中一定要提一下，當中兩位前輩是台灣人呢，WOOW
Chien-Yao Wang和Hong-Yuan Mark Liao先生他們也是YOLOR的作者呢!!
是不是很光榮呀🎊🎊🎊

簡單介紹
====
在今年 2022 的 7 月，YOLOV7 正式閃亮登場啦~，它在5 FPS - 160 FPS 的範圍內，速度(speed)跟準確度(accuracy)是目前即時物件偵測(Real-time Object Detection)相關技術中的第一名，意即我最快也最準。
![image](https://user-images.githubusercontent.com/46515944/179974124-42d52bbf-bc61-4e7b-adef-e1bbf22c379a.png)

而 YOLOv7 一文的主要貢獻有以下四點：

1. 藉由設計幾種 bag-of-freebies(一種優化物件偵測的技巧，2019年由李沐大神所提出，論文請點我) 的方法，在不增加推理成本(inference cost)的前提下，來提升物件偵測的準確度。
2. 解決兩個物件偵測演進的問題：
(一)如何重新參數化取代原始模組；
(二)動態標籤分配如何處理分配給不同輸出層的問題
3. 利用 “extend” , “compound scaling” 的方式，有效地使用參數和計算
4. YOLOV7 可以有效減少 SOTA (State-of-the art，最先進的) 即時物件偵測 約40%的參數及50%的計算量，來提高速度與準確度。

💎主要教學從以下開始💎
====
## 🌱首先，請先至github下載檔案

如果直接在 Google 上搜尋，排行第一個可能是 jinfagang (有興趣點我)，
我起初是使用他的版本，但後面遇到不少問題後，才跑回去看論文內原作所 提供的網址，操作起來就流暢多了，這邊沒有要戰誰好，只是單純依個人在操作上的偏好，有興趣者可以點擊上方的超連結，而下方的連結為原作王建饒博士的Github。若要跟著以下的步驟進行訓練，請下載下面連結的程式碼

> 👉 https://github.com/WongKinYiu/yolov7

![image](https://user-images.githubusercontent.com/46515944/179974399-071955aa-3d58-4b4e-8759-872dcdc308c9.png)

##🌱下滑找到 Performance下載任一模型的預訓練檔，要挑哪一個就根據自己的需求啦(檔名為 .pt)，之後會用在訓練上

![image](https://user-images.githubusercontent.com/46515944/179974423-a5fa4b92-78ca-442e-83ff-cff0f0822121.png)

##🌱再來，打開Anaconda Prompt，並且新建一個環境，這邊就省略教學啦~
如果不知道要怎麼建立的話，可以看我的Github 👇 https://github.com/huchi00057/YOLOX

![image](https://user-images.githubusercontent.com/46515944/179974451-7f3727c4-f041-44e0-9ba7-b9eb72e135d5.png)

而本次我是建立一個Python 版本為3.9的環境，名為yolov7。

##🌱Anaconda Prompt 切到 最一開始創的yolov7環境
並且切到 yolov7資料夾

![image](https://user-images.githubusercontent.com/46515944/179974570-5f6f33fb-1a37-46f7-a885-76d00a5bb5d7.png)

##🌱接著安裝套件：

    pip install requirement.txt

如果你在安裝時還缺其他套件的話，再另外搜尋下載嚕~~
😙 如果你要用GPU去訓練，要特別去下載Pytorch套件優，不會就 點我

##🌱基本上安裝到這裡就差不多了，我們可以來測試安裝行不行啦~~

    python detect.py — weights ✍️yolo7.pt(前面下載的預訓練檔) — source inference/images/horses.jpg(你資料夾一定有)

跑完後的結果會存在 yolov7/runs/detect/exp? 內，如下圖。
長這樣，代表你裝好了，棒棒很棒超棒👏👏

![image](https://user-images.githubusercontent.com/46515944/179974632-13d81380-1ca9-44a5-b527-10283863e18a.png)


一切待續後，準備來訓練模型囉~~
##🌱首先，先準備好資料集，目前我只會用YOLO格式(.txt)的資料集去做訓練，如果你本來已經有資料集了(Pascal VOC 或 COCO格式)，請自行轉檔。

【XUANㄟ教學文章】yolov7Pascal VOC .xml COCO .json YOLO .txt
我本來就有以上兩種格式，但在訓練yolor的時候碰壁，找了好久，總算找到一篇有用的文章：https://hackmd.io/@jim93073/r1laqq0jF
(我只有改CLASS，然後再執行main.py，
像這樣：python main.py — path D:\yolox\data\1\test.txt — output test.json)

![image](https://user-images.githubusercontent.com/46515944/179974673-b408f3d7-4fe2-4843-b943-5fa3f149b0c3.png)

![image](https://user-images.githubusercontent.com/46515944/179974686-c796b5c6-bfd0-47e1-9bfb-25709febc60a.png)

這邊給大家參考一下，我的資料擺放~

總結資料集的準備，你需要以下：

**image資料夾：圖片檔(.jpg)、標記檔(.txt)**
**兩個.txt檔案：所有訓練圖片的記事本還有驗證檔的(train.txt , valid.txt)**

你一定會問，我就不會才要看你文章，你還不教我怎麼標記圖片，這樣對嗎 All right. 我大概說明一下齁，很簡單

    #下載標記圖片的工具叫做 labelim
    pip install labelimg
    #開啟它
    labelimg
![image](https://user-images.githubusercontent.com/46515944/179975225-5d9c86c8-e3f4-40c0-ba27-cb550ddd2d32.png)
![image](https://user-images.githubusercontent.com/46515944/179975237-73eb149b-e933-4d29-95f3-7bdf0a6bcdef.png)

開啟充滿要拿來訓練圖片的資料夾
![image](https://user-images.githubusercontent.com/46515944/179975260-9298014c-1aa3-4b7b-967f-cdebd1d0660b.png)
![image](https://user-images.githubusercontent.com/46515944/179975275-48c0cff3-0d07-4917-a902-2632b2ba536a.png)

先點右邊倒數第四個按鈕(YOLO)，按個兩下，讓它圖案保持跟一開始一樣，再點擊右邊的 Create RectBox 將物件框選起來，並輸入類別名稱，最後按save進行儲存(儲存格是要是.txt，如果不是就按倒數第四個按鈕)，以此類推其他圖片(Next Image 鈕跳到下一張圖片)。全部完成後關掉即可。

❌類別可不可以用中文，我不知道，你試試看吧，我不想冒debug的風險，所以沒有嘗試哈哈


## 🌱在跑模型前的最後一個步驟就是修改參數

到 cfg 資料夾找你適合想要的 .yaml 複製到...你記得的地方，並更改文件名稱
![image](https://user-images.githubusercontent.com/46515944/179975561-af92ef57-d230-43dd-9ae4-79ab7984dfb4.png)

我個人是放在cfg資料夾裡，並更名為 colon.yaml，另外更改 nc ：3，因為我分成三類別。
![image](https://user-images.githubusercontent.com/46515944/179975594-827674fb-6aaf-4f39-8cc1-46bf6af3cbd5.png)

## 🌱接著複製 data 資料夾的colon.yaml檔，修改訓練集(train)與驗證集(val)的路徑與類別名稱(names)及數量(nc)。
![image](https://user-images.githubusercontent.com/46515944/179975638-def2b7ae-543a-4199-9afb-9691f1e3ab90.png)

一切待續後，真的要來訓練囉~~

#  🌱輸入訓練指令

    python train.py — device 0 — batch-size 4 — data data/colon.yaml — cfg cfg/colon.yaml

(因為我有用 gpu 去跑，所以有加 — device 0， — epoch 可以根據自己的需求做調整，預設是300)

開始訓練後會出現
![image](https://user-images.githubusercontent.com/46515944/179975708-1e66e333-489c-459f-a4db-70562cd29553.png)

跑完長這樣
![image](https://user-images.githubusercontent.com/46515944/179975742-9a8a90f2-1afc-4f5f-a920-75d57e2c49a0.png)

## 🌱歷經15小時的訓練，總算可以來收割啦~

> 以下指令是用來跑圖片或影片的
    python detect.py — weights D:\desktop\yolov7\runs\train\exp8\weights\best.pt — source test(我充滿圖片的資料夾)

在runs/detect資料夾中會出現你try的圖片上附加預測框框。

## 🌱另外，在輸出的資料夾中(runs/runs/exp)，內容物包含權重檔，f1.pr.r.p曲線圖，cmd 的紀錄以及訓練過程輸出的預測樣子(被我擋住的圖片)。

![image](https://user-images.githubusercontent.com/46515944/179975828-8e3e1577-59eb-4484-987a-71dc326224e5.png)

## 🌱假如訓練到一半斷掉了想要接續訓練，只需更改預訓練檔即可
    python train.py — device 0 — batch-size 4 — data data/colon.yaml — cfg cfg/colon.yaml — weights 跑到一半的權重.pt

🤓🤓以上是YOLOV7的訓練教學，由於圖片涉及隱私，所以很多小細節就不另外放圖片展示。若遇到任何問題，可進一步再底下留言⌨⌨討論

🔗資料參考：(https://www.youtube.com/watch?v=ag88beS_fvM)
(如果有些步驟還是不曉得怎麼弄，可以參考這個影片)
