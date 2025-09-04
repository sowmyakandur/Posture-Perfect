import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('yoga_pose_model.h5')

class_names = {0: 'bridge', 1: 'cat-cow', 2: 'child', 3: 'cobra'}

cap = cv2.VideoCapture(0)

new_width = 224
new_height = 224

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, (new_width, new_height))

    img = np.expand_dims(resized_frame, axis=0)
    img = img / 255.0

    preds = model.predict(img)
    max_confidence = np.max(preds)
    class_idx = np.argmax(preds)
    class_name = class_names[class_idx]

    if max_confidence > 0.5:
        cv2.putText(frame, class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
    else:
        cv2.putText(frame, "No pose detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    2, cv2.LINE_AA)

    cv2.imshow('Yoga Pose Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()