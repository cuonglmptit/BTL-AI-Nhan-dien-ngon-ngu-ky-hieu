# với test thì gần giống dataCollector nên ta sẽ copy lại và sửa đôi chút, đồng thời xóa bớt comment để cho dễ nhìn
# ta chỉ cần thêm phần lấy label(nhãn) và cho vào 1 list để khi predict là nhãn nào thì sẽ in ra và hình mà xuát ra thì xóa phần khung xương đi cho thân thiện người dùng
#chỉ cần lấy imgData để so sánh dự đoán ra nhãn nào
import cv2
import numpy as np
import math
from cvzone.HandTrackingModule import HandDetector  # thư viện dùng để xác định tay
from cvzone.ClassificationModule import Classifier # thư viện dùng để xác định phân loại bằng thống kê model mà ta tạo được

# hàm draw_text để chữ có nền cho dễ nhìn, ko cần cũng được
def draw_text(img, text,
          font=cv2.FONT_ITALIC,
          pos=(0, 0),
          font_scale=3,
          font_thickness=2,
          text_color=(255, 255, 255),
          text_color_bg=(0, 0, 0)
          ):

    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, (x,y-5), (x + text_w+3, y +10+ text_h), text_color_bg, -1)
    cv2.putText(img, text, (x+3, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return text_size


cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY, params=[cv2.CAP_PROP_FRAME_WIDTH, 1280,cv2.CAP_PROP_FRAME_HEIGHT, 720])
# cap = cv2.VideoCapture('ASL.mp4')
detector = HandDetector(maxHands=2)
#tạo 1 phân loại từ thư viện class, đọc vào là model keras_model.h5 ta tạo được từ web https://teachablemachine.withgoogle.com/train/image,
# phân loại nhãn là file labels.txt
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

#tạo các nhãn để khi xác định được thì sẽ in ra
labels = []
f = open('Model/labels.txt', 'r')
while True:
    line = f.readline()
    if line != '':
        txt = line[line.find(' ')+1:].strip()
        labels.append(txt)
    else:
        break
f.close()
# print(labels)

offset = 20
imgSize1Hand = 300
imgSize2Hand = 600
# Vòng lặp liên tục để nhận ảnh vào và show ảnh
while True:
    success, img = cap.read()

    img = cv2.flip(img, 1)

    # tạo 1 ảnh khác là output, ảnh này chỉ vẽ hình chữ nhật quanh vùng tay và hiện chữ chứ không hiện khung xương
    imgOutput = img.copy()
    # tạo 4 biến để vẽ hình chữ nhật quanh vùng xác định được
    xTrai = 0
    yTren = 0
    xPhai = 0
    yDuoi = 0

    hands, img = detector.findHands(img, True, False)

    if hands:
        try:
            imgData = np.ones((imgSize1Hand, imgSize1Hand, 3), np.uint8) * 255
            imgData2Hand = np.ones((imgSize2Hand, imgSize2Hand, 3), np.uint8) * 255

            if len(hands) == 1:
                hand = hands[0]
                x, y, w, h = hand['bbox']
                imgCropHand = img[y - offset:y + h + offset, x - offset:x + w + offset]
                # cv2.imshow("Img Crop", imgCropHand)

                imgData = np.ones((imgSize1Hand, imgSize1Hand, 3), np.uint8) * 255

                aspectRatio = h / w
                if aspectRatio > 1:
                    k = imgSize1Hand / h
                    wCal = math.ceil(k * w)  # vì là giá trị có thể lẻ nên phải lấy cận trên
                    wGap = math.ceil((imgSize1Hand - wCal) / 2)
                    imgResize = cv2.resize(imgCropHand, (wCal, imgSize1Hand))
                    imgData[:, wGap:wCal + wGap] = imgResize
                else:  # nếu chiều width mà lớn hơn chiều height => aspectRatio < 1 và khi fit thì w = imgSize1Hand
                    k = imgSize1Hand / w
                    hCal = math.ceil(k * h)
                    hGap = math.ceil((imgSize1Hand - hCal) / 2)
                    imgResize = cv2.resize(imgCropHand, (imgSize1Hand, hCal))
                    imgData[hGap:hGap + hCal, :] = imgResize

                # gán tọa đổ vẽ hình chữ nhật cho trường hợp 1 tay
                # vị trí vẽ bắt đầu từ tọa độ (x, y) là điểm gốc của khung xương(ở góc trái trên), đến toạ độ (x+w, y+h) chính là điểm cuối(góc phải dưới)
                xTrai = x
                yTren = y
                xPhai = x+w
                yDuoi = y+h

            # trường hợp 2 bàn tay
            elif len(hands) == 2:
                handLeft = {}
                handRight = {}

                if hands[0]['type'] == 'Left' and hands[1]['type'] == 'Right':
                    handLeft = hands[0]
                    handRight = hands[1]
                elif hands[1]['type'] == 'Left' and hands[0]['type'] == 'Right':
                    handLeft = hands[1]
                    handRight = hands[0]

                xL, yL, wL, hL = handLeft['bbox']
                xR, yR, wR, hR = handRight['bbox']

                imgCropHandLeft = img[yL - offset:yL + hL + offset, xL - offset:xL + wL + offset]
                imgCropHandRight = img[yR - offset:yR + hR + offset, xR - offset:xR + wR + offset]
                # cv2.imshow("Img Crop Hand 1", imgCropHandLeft)
                # cv2.imshow("Img Crop Hand 2", imgCropHandRight)

                imgData = np.ones((imgSize1Hand, imgSize2Hand, 3), np.uint8) * 255

                # Fit tay trái vào imgData trước, tay trí thì làm giống 1 tay vì tay trái nằm bên trái imgData
                aspectRatioL = hL / wL
                if aspectRatioL > 1:
                    k = imgSize1Hand / hL
                    wCal = math.ceil(k * wL)
                    wGap = math.ceil((imgSize1Hand - wCal) / 2)
                    imgResizeL = cv2.resize(imgCropHandLeft, (wCal, imgSize1Hand))
                    imgData[:, wGap:wCal + wGap] = imgResizeL
                else:
                    k = imgSize1Hand / wL
                    hCal = math.ceil(k * hL)
                    hGap = math.ceil((imgSize1Hand - hCal) / 2)
                    imgResizeL = cv2.resize(imgCropHandLeft, (imgSize1Hand, hCal))
                    imgData[hGap:hGap + hCal, :imgSize1Hand] = imgResizeL
                # Fit tay phải vài imgData, giống tay trái nhưng vị trí bắt đầu của cột sẽ là imgSize1Hand
                aspectRatioR = hR / wR
                if aspectRatioR > 1:
                    k = imgSize1Hand / hR
                    wCal = math.ceil(k * wR)
                    wGap = math.ceil((imgSize1Hand - wCal) / 2)
                    imgResizeR = cv2.resize(imgCropHandRight, (wCal, imgSize1Hand))
                    imgData[:, imgSize1Hand + wGap:imgSize1Hand + wCal + wGap] = imgResizeR
                else:
                    k = imgSize1Hand / wR
                    hCal = math.ceil(k * hR)
                    hGap = math.ceil((imgSize1Hand - hCal) / 2)
                    imgResizeR = cv2.resize(imgCropHandRight, (imgSize1Hand, hCal))
                    imgData[hGap:hGap + hCal, imgSize1Hand:imgSize2Hand] = imgResizeR

                # gán tọa độ vẽ hình chữ nhật cho trường hợp 2 tay
                # ta phải hình dung hình chữ nhật sẽ bao quanh cả 2 tay nên ta phải tìm tọa độ và độ dài chính xác để vẽ (tọa độ là dòng, cột bắt đầu-kết thúc)
                # trường hợp này ta phải tìm toạ độ (trái trên) và (dưới phải) từ S = {xL, yL, wL, hL, xR, yR, wR, hR}
                # Tọa độ trái trên: (min(X), min(Y))
                # Tọa độ phải dưới: (max(xL+wL, xR+wR), max(yL+hL, yR+hR))
                xTrai = min(xL, xR)
                yTren = min(yL, yR)
                xPhai = max(xL+wL, xR+wR)
                yDuoi = max(yL+hL, yR+hR)
                imgData2Hand[150:imgSize1Hand + 150, :] = imgData
                imgData = cv2.resize(imgData2Hand, (0, 0), fx=0.5, fy=0.5)

            # vẽ hình chữ nhật đồng thời thêm 1 tý offset cho to ra
            cv2.rectangle(imgOutput, (xTrai - offset, yTren - offset), (xPhai + offset, yDuoi + offset), (0,252,124), 1)
            #xác định phân loại ảnh imgData là label nào
            prediction, index = classifier.getPrediction(imgData)
            # print(max(prediction))
            # print(index)
            accPercent = round(max(prediction)*100, 2)
            # cv2.putText(imgOutput, labels[index], (xTrai - offset-5, yTren - offset-5), cv2.FONT_ITALIC, 1,  (125, 55, 255), 3)
            # cv2.putText(imgOutput, str(accPercent)+'%', (xTrai - offset, yTren - offset + 50), cv2.FONT_ITALIC, 1,(255,255,0), 3)
            draw_text(imgOutput, labels[index], font_scale=1, pos=(xTrai - offset, yTren - 4*offset-13),text_color_bg=(125, 55, 255), font_thickness=3)
            draw_text(imgOutput, str(accPercent)+'%', font_scale=1, pos=(xTrai - offset, yTren - 3*offset+6), text_color_bg=(204,204,0), font_thickness=3)
            cv2.imshow("Img Data", imgData)
        except:
            print("Co loi xac dinh tay")
            # cv2.putText(imgOutput, "KHONG THE PHAT HIEN!", (0, 2 * offset), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 250), 3)
            draw_text(imgOutput, "KHONG THE PHAT HIEN!", font_scale=1, pos=(0,5), text_color_bg=(0, 0, 250), font_thickness=3)
    # imgOutput = cv2.resize(imgOutput, (0, 0), fx=1.5, fy=1.5)
    cv2.imshow("Img", imgOutput)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
