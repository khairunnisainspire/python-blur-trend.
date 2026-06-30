import cv2
import mediapipe as mp

# Inisialisasi MediaPipe untuk deteksi tangan
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Mengakses webcam dan mengatur resolusi agar tidak glitch
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Webcam tidak terdeteksi.")
        break
        
    # Membalik gambar agar seperti cermin
    frame = cv2.flip(frame, 1)

    # Memproses deteksi tangan
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    peace_sign_detected = False

    # Mengecek gaya 2 jari (Peace)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            index_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y
            middle_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y
            ring_down = hand_landmarks.landmark[16].y > hand_landmarks.landmark[14].y
            pinky_down = hand_landmarks.landmark[20].y > hand_landmarks.landmark[18].y
            
            if index_up and middle_up and ring_down and pinky_down:
                peace_sign_detected = True
                break 

    # JIKA gaya 2 jari terdeteksi, blur SELURUH layar
    if peace_sign_detected:
        # Terapkan blur langsung ke variabel 'frame' secara utuh
        # Angka (99, 99) bisa dibesarkan lagi (misal 151, 151) kalau mau lebih blur!
        frame = cv2.GaussianBlur(frame, (10, 10), 0)

    # Menampilkan hasil video
    cv2.imshow('Tren Foto Kita Blur - 2 Jari', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Membersihkan memori
cap.release()
cv2.destroyAllWindows()