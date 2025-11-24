import cv2
import face_recognition


def main():
    # 打开电脑摄像头（0表示默认摄像头，多个摄像头可尝试1、2等）
    video_capture = cv2.VideoCapture(0)

    # 加载已知人脸（可添加多个人脸）
    # 第一张已知人脸（修正引号：单引号或双引号均可，但不要嵌套）
    known_image1 = face_recognition.load_image_file("D:\\机器学习\\人脸识别\\人脸图片\\微信图片_20240522215634.jpg")  # 注意Windows路径用双反斜杠
    known_face_encoding1 = face_recognition.face_encodings(known_image1)[0]

    # 第二张已知人脸（新增）
    known_image2 = face_recognition.load_image_file("D:\\机器学习\\人脸识别\\人脸图片\\微信图片_20240523094610.jpg")
    known_face_encoding2 = face_recognition.face_encodings(known_image2)[0]

    # 存储所有已知人脸的特征和名称（按顺序对应）
    known_face_encodings = [known_face_encoding1, known_face_encoding2]
    known_face_names = ["贾常辉", "贾常辉"]  # 对应上面的人脸名称

    # 存储已检测到的人脸特征和名称
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True  # 控制每帧是否处理（提升性能）

    while True:
        # 读取摄像头画面
        ret, frame = video_capture.read()
        if not ret:
            print("无法获取摄像头画面")
            break

        # 缩小画面尺寸以提升处理速度（1/4尺寸）
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        # 转换颜色空间（OpenCV用BGR，face_recognition用RGB）
        rgb_small_frame = small_frame[:, :, ::-1]

        # 每隔一帧处理一次（减少计算量）
        if process_this_frame:
            # 检测画面中所有人脸的位置和特征
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # 对比所有已知人脸（tolerance越小越严格，默认0.6）
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                name = "Unknown"  # 默认为未知

                # 如果匹配到已知人脸，则使用对应名称
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame  # 切换处理状态

        # 在画面上绘制人脸框和名称
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # 还原人脸位置到原始画面尺寸（因为之前缩小了1/4）
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # 绘制人脸矩形框（绿色，线宽2）
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            # 绘制名称背景框
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            # 绘制名称文字（白色，字体大小0.75）
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.75, (255, 255, 255), 1)

        # 显示处理后的画面
        cv2.imshow('Face Recognition', frame)

        # 按 'q' 键退出程序
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放资源
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()