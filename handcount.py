import cv2
import mediapipe as mp

# Initialize Mediapipe for hand tracking
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils  # Utility for drawing hand landmarks

# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True
while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for natural (mirror-like) interaction
    frame = cv2.flip(frame, 1)

    # Convert the frame from BGR to RGB
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark predictions
    result = hands.process(framergb)

    # Post-process the result
    if result.multi_hand_landmarks:
        for handslms in result.multi_hand_landmarks:
            # Draw hand landmarks on the frame
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Get coordinates of fingers
            tip_ids = [4, 8, 12, 16, 20]  # Tip landmarks for thumb, index, middle, ring, and pinky fingers
            finger_count = sum([1 for id in tip_ids if handslms.landmark[id].y < handslms.landmark[id - 2].y])

            # Display the number of extended fingers
            cv2.putText(frame, f"Fingers: {finger_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Determine number based on finger count (customize this logic)
            if finger_count == 0:
                number = 0
            elif finger_count == 1:
                number = 1
            elif finger_count == 2:
                number = 2
            elif finger_count == 3:
                number = 3
            elif finger_count == 4:
                number = 4
            elif finger_count == 5:
                number = 5


            # Display the identified number
            cv2.putText(frame, f"Number: {number}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Number Identification", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
