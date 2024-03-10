from tensorflow.keras.models import load_model # takes only one input as feed and expects one output
from tensorflow.nn import softmax
import numpy as np
from tensorflow.keras import preprocessing
import cv2
import os

def predict(image_path):
    classifier_model = "alkaline-model.h5"
      
    model = load_model(classifier_model, compile=False)
            
    test_image_load = cv2.imread(image_path)
    test_image = cv2.resize(test_image_load, (256,256), interpolation=cv2.INTER_LINEAR)
    test_image = preprocessing.image.img_to_array(test_image)
    test_image = test_image / 255.0
    test_image = np.expand_dims(test_image, axis=0)
    predictions = model.predict(test_image)
    print(predictions)
    label = 'Energizer' if predictions < 0.5 else 'Publix'
    return label

def save_frame_camera_key(device_num, dir_path, basename, ext='jpg', delay=1, window_name='frame'):
    cap = cv2.VideoCapture(device_num)

    if not cap.isOpened():
        return

    os.makedirs(dir_path, exist_ok=True)
    base_path = os.path.join(dir_path, basename)

    n = 0
    while True:
        ret, frame = cap.read()
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key == ord('c'):
            fname = '{}_{}.jpg'.format(base_path, n,)
            cv2.imwrite(fname, frame)
            predict(fname)
    
            n += 1
        elif key == ord('q'):
            break

    cv2.destroyWindow(window_name)

if __name__ == "__main__":
    img1 = 'user-data/test1.jpg'
    img2 = 'user-data/test2.jpg'
    print(predict(img1))
    print(predict(img2))
    # print(predict(img_resize))
    # save_frame_camera_key(0, 'data', 'img')
