import cv2
import numpy as np
import math
import time
from cvzone.HandTrackingModule import HandDetector  # thư viện dùng để xác định tay

# chọn camera để thực hiện ghi hình, tham số 0 là id camera mặc định của máy
cap = cv2.VideoCapture(0, apiPreference=cv2.CAP_ANY, params=[cv2.CAP_PROP_FRAME_WIDTH, 1280,cv2.CAP_PROP_FRAME_HEIGHT, 720])
# cap = cv2.VideoCapture(0)
# HandeTector(maxHands=2) => số bàn tay có thể xuất hiện trong hình
detector = HandDetector(maxHands=2)

offset = 20  # padding cho phần chọn hình to hay nhỏ
# tạo khung cố định thu thập cho trường hợp 1 tay được phát hiện và 2 tay được phát hiện, với 1 tay thì sẽ cho độ lớn ma trận ảnh là 300, còn với 2 tay thì sẽ cho là 600
# tuy nhiên với trường hợp 2 tay thì hình sẽ là ghép của 2 hình, do vậy chỉ có chiều width là 600 còn chiều height vẫn là 300
imgSize1Hand = 300
imgSize2Hand = 600
#đường dẫn cho nhãn mà ta muốn tạo data
folder = "Data/Train data ver2/A"
counter = 0
# Vòng lặp liên tục để nhận ảnh vào và show ảnh
while True:
    # cap.read() trả về 2 giá trị, đầu tiên là có nhận được ảnh không và 2 là ma trận ảnh của cảnh vừa ghi được
    success, img = cap.read()

    # ta sẽ lật dọc ảnh để thiện thao tác, vì ngôn ngữ kí hiệu có thể áp dụng hiệu ứng
    # ảnh phản chiếu (mirror image) cho nên việc tay trái hay tay phải sẽ không tác động đến việc signing(biểu diễn ngôn ngữ)
    img = cv2.flip(img, 1)

    # ta sẽ cho hands lấy giá trị thông số bàn tay xác định được trong ảnh và img sẽ được vẽ khung xương cho tay trong ảnh
    # hands sẽ là list có 1 phần tử nếu 1 tay được phát hiện, 2 phần tử nếu 2 tay được phát hiện, max là 2 tay vì ta đã đặt số tay max là 2
    hands, img = detector.findHands(img, True, False)

    # ta sẽ lấy thông số 'bbox' (border box) sẽ là vij trí cột (x), vị tri dòng(y), width, height của bàn tay xác định bởi detector.findHands()
    # tuy nhiện có 2 trường hợp là có 1 tay và có 2 tay thì sẽ phải xử lý với tay thứ nhất là hands[0] và tay thứ 2 là hands[1]
    # nếu có bàn tay xuất hiện trong hình, ta sẽ thực hiện việc lấy dữ liệu
    if hands:
        try:
            imgData = np.ones((imgSize1Hand, imgSize1Hand, 3), np.uint8) * 255
            imgData2Hand = np.ones((imgSize2Hand, imgSize2Hand, 3), np.uint8) * 255

            # trường hợp chỉ có 1 bàn tay
            if len(hands) == 1:
                # lấy thông số của bàn tay
                hand = hands[0]
                x, y, w, h = hand['bbox']
                # tạo 1 ảnh cắt phần tay khung xương ra từ ảnh img đã được vẽ khung xương
                # thêm offset để lấy thêm phần ngoài một tý để được hình toàn bàn tay (dòng bắt đầu giảm đi:dòng kết thúc tăng lên) và (cột bắt đầu giảm đi:cột kết thúc tăng lên)
                imgCropHand = img[y - offset:y + h + offset, x - offset:x + w + offset]
                # cv2.imshow("Img Crop", imgCropHand)

                # tạo 1 khung ảnh dữ liệu train với thông số của 1 bàn tay là 300, với nền là toàn màu trắng thể loại numpyarray, số kênh là 3
                # việc tạo ảnh có thông số xác định giúp cho việc train chính xác hơn
                imgData = np.ones((imgSize1Hand, imgSize1Hand, 3), np.uint8) * 255
                # Fit ảnh imgCrop1Hand vào imgData1Hand và căn giữa (do ảnh khung xướng cắt ra có thể to nhỏ khác nhau nên cần phải cố định vào ảnh imgData)
                # tỉ lệ ảnh
                aspectRatio = h / w
                # nếu chiều height mà lớn hơn chiều width => aspectRatio > 1 và khi fit thì h = imgSize1Hand
                if aspectRatio > 1:
                    # tính wCal là khoảng chiều ngang của ảnh khi fit chiều cao ảnh (h/w = imgSize1Hand/wCal) => wCal = (imgSize1Hand*w)/h
                    k = imgSize1Hand / h
                    wCal = math.ceil(k * w)  # vì là giá trị có thể lẻ nên phải lấy cận trên
                    # tính khoảng cách bị thừa chiều ngang khi fit ảnh
                    wGap = math.ceil((imgSize1Hand - wCal) / 2)
                    # resize ảnh cắt được và ghép vào imgData
                    imgResize = cv2.resize(imgCropHand, (wCal, imgSize1Hand))
                    imgData[:, wGap:wCal + wGap] = imgResize
                else:  # nếu chiều width mà lớn hơn chiều height => aspectRatio < 1 và khi fit thì w = imgSize1Hand
                    # cách tính như phần trên mà ngược lại w thành h
                    k = imgSize1Hand / w
                    hCal = math.ceil(k * h)
                    hGap = math.ceil((imgSize1Hand - hCal) / 2)
                    imgResize = cv2.resize(imgCropHand, (imgSize1Hand, hCal))
                    imgData[hGap:hGap + hCal, :] = imgResize

            # trường hợp 2 bàn tay
            elif len(hands) == 2:
                # lấy thông số của bàn tay 1 và 2
                handLeft = {}
                handRight = {}
                # gán cho đúng tay trái và phải để tý nữa ghép vào imgData không bị nhầm bên
                if hands[0]['type'] == 'Left' and hands[1]['type'] == 'Right':
                    handLeft = hands[0]
                    handRight = hands[1]
                elif hands[1]['type'] == 'Left' and hands[0]['type'] == 'Right':
                    handLeft = hands[1]
                    handRight = hands[0]

                xL, yL, wL, hL = handLeft['bbox']
                xR, yR, wR, hR = handRight['bbox']
                # tạo 1 ảnh cắt phần tay khung xương ra từ ảnh img đã được vẽ khung xương
                # thêm offset để lấy thêm phần ngoài một tý để được hình toàn bàn tay (dòng bắt đầu giảm đi:dòng kết thúc tăng lên) và (cột bắt đầu giảm đi:cột kết thúc tăng lên)
                imgCropHandLeft = img[yL - offset:yL + hL + offset, xL - offset:xL + wL + offset]
                imgCropHandRight = img[yR - offset:yR + hR + offset, xR - offset:xR + wR + offset]
                # cv2.imshow("Img Crop Hand 1", imgCropHandLeft)
                # cv2.imshow("Img Crop Hand 2", imgCropHandRight)

                # tạo 1 khung ảnh dữ liệu train cho 2 bàn tay
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
                    # khác phần này vì chiều ngang chỉ cho ghép đến 1 nữa (imgSize1Hand=300)
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
                imgData2Hand[150:imgSize1Hand+150, :] = imgData
                imgData = cv2.resize(imgData2Hand,(0,0),fx=0.5,fy=0.5)
            cv2.imshow("Img Data", imgData)
            if cv2.waitKey(1) == ord(' '):
                counter += 1
                cv2.imwrite(f'{folder}/Image_{time.time()}.png', imgData)
                print(counter)
        except:
            print("Co loi tay ra khoi man hinh")
    cv2.imshow("Img", img)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
