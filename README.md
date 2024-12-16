# eye-blink-game

This Python script is a **real-time face mask detection application** using **TensorFlow** for model inference and **OpenCV** for video capture and face detection.

---

### Features:
1. **Convolutional Neural Network (CNN) Architecture**:
   - Built using `tensorflow.keras.Sequential` with the following layers:
     - **2 Conv2D layers**: 100 filters, `(3,3)` kernel, ReLU activation.
     - **2 MaxPooling2D layers**: Pool size `(2,2)`.
     - **Flatten layer**: Converts 2D feature maps into a 1D array.
     - **Dropout layer**: With a dropout rate of `0.5` to reduce overfitting.
     - **3 Dense layers**: Fully connected layers, including the output layer.
   - The output layer predicts two classes (`Mask` or `No Mask`).

   - **Compilation**:
     - Optimizer: **Adam**
     - Loss: **binary crossentropy** (used for binary classification).
     - Metric: **accuracy**.

2. **Face Detection**:
   - Uses Haar Cascade (`haarcascade_frontalface_default.xml`) for detecting faces in real-time from the webcam feed.

3. **Real-time Mask Prediction**:
   - For each detected face:
     1. Extracts the region of interest (ROI).
     2. Resizes the ROI to `150x150`.
     3. Normalizes the pixel values (scales them to [0,1]).
     4. Passes the ROI through the CNN to predict the presence of a mask.
     5. Draws a rectangle around the face with:
        - **Red** rectangle for "No Mask".
        - **Green** rectangle for "Mask".
     6. Displays the prediction label (`Mask` or `No Mask`) above the rectangle.

4. **Video Stream**:
   - Captures live video feed using `cv2.VideoCapture(0)`.
   - Flips the video horizontally for a mirror effect.

5. **Exit Mechanism**:
   - Press the **Esc** key to exit the application and release the webcam.

---

### Application Flow:
1. **Model Prediction**:
   - Extracts each detected face.
   - Preprocesses it to match the input requirements of the CNN.
   - Classifies the face as **Mask** or **No Mask** using the model.

2. **Visual Feedback**:
   - The application overlays rectangles and labels over detected faces:
     - Green for "Mask".
     - Red for "No Mask".

3. **Output Display**:
   - Opens a window (`LIVE`) that displays the live video with visual annotations.

---

### Libraries Used:
- **TensorFlow/Keras**: For building and running the CNN model.
- **OpenCV**: For video feed capture, face detection, and real-time visualization.
- **NumPy**: For image preprocessing.

---

### Notes:
1. **Pre-trained Model**:
   - The `cnn` model is created in the script but lacks training code. For production use, the model should be trained on a labeled dataset of face images with/without masks.
   - Alternatively, load a pre-trained model using `load_model()`.

2. **Performance Considerations**:
   - Ensure proper lighting conditions for better face detection.
   - Haar cascades are computationally lightweight but may not perform well in challenging environments.

3. **Improvements**:
   - Use a pre-trained face detection model (e.g., **DNN-based face detector** or **MTCNN**) for more robust detection.
   - Incorporate a pre-trained deep learning model (e.g., MobileNetV2) for better mask detection accuracy.
