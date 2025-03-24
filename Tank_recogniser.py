import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from IPython.display import Video
from ultralytics import YOLO

class Tank_recogniser:
    def __init__(self, model_path="yolov11n.pt", conf_threshold=0.2):
        """
        Initialize the Tank_recogniser class.

        Args:
            model_path (str): Path to the YOLO model file.
            conf_threshold (float): Confidence threshold for filtering detections.
        """
        # FileNotFoundError
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold  # Мінімальний поріг впевненості

    def _resize_and_pad(self, image_path, target_size=640):
        """
        Resize the image while maintaining aspect ratio and adding black padding to fit the target size.

        Args:
            image_path (str): Path to the input image.
            target_size (int): The desired size for the image.

        Returns:
            tuple: Processed image, scale factor, left padding, top padding, original image shape.
        """
        img = cv2.imread(image_path)  # Завантажуємо BGR-зображення
        
        h, w = img.shape[:2]
        scale = min(target_size / w, target_size / h)

        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        delta_w = target_size - new_w
        delta_h = target_size - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        return padded_img, scale, left, top, (w, h)

    def _rescale_detections(self, detections, scale, left, top, original_size):
        """
        Rescale detections back to the original image size.

        Args:
            detections (ultralytics.engine.results.Boxes): Detected objects in resized image.
            scale (float): Scale factor used for resizing.
            left (int): Left padding added during resizing.
            top (int): Top padding added during resizing.
            original_size (tuple): Original image dimensions (width, height).

        Returns:
            list: Rescaled detections [(x1, y1, x2, y2, conf, cls), ...]
        """
        original_w, original_h = original_size
        new_detections = []

        for box in detections:
            x1, y1, x2, y2 = box.xyxy[0].tolist()  
            conf = float(box.conf[0])  
            cls = int(box.cls[0]) 
            x1 = max((x1 - left) / scale, 0)
            y1 = max((y1 - top) / scale, 0)
            x2 = min((x2 - left) / scale, original_w)
            y2 = min((y2 - top) / scale, original_h)

            new_detections.append([x1, y1, x2, y2, conf, cls])

        return new_detections

    def _draw_detections(self, image, detections):
        """
        Draw bounding boxes with class labels on an image.

        Args:
            image (numpy.ndarray): Original image.
            detections (list): List of rescaled detections [(x1, y1, x2, y2, confidence, class_id), ...].

        Returns:
            numpy.ndarray: Image with drawn detections.
        """
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            label = f"{self.model.names[cls]} {conf:.2f}"
            image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 3)
            image = cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        return image

    def predict_image(self, image_path, target_size=640, save=False , show=False):
        """
        Run YOLO prediction on an image with preprocessing and rescaling.

        Args:
            image_path (str): Path to the input image.
            target_size (int): The target size for YOLO detection.
            show (bool): Whether to display the final image with detections.
            save (bool or str):  Whether to save the final image with detections.
        """
        padded_img, scale, left, top, original_size = self._resize_and_pad(image_path, target_size)
              
        results = self.model.predict(padded_img, conf=self.conf_threshold)
        detections = results[0].boxes 
        rescaled_detections = self._rescale_detections(detections, scale, left, top, original_size)

        original_img = cv2.imread(image_path)
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB) 

        original_img = self._draw_detections(original_img, rescaled_detections)

        if show:
            plt.figure(figsize=(10, 10))
            plt.imshow(original_img)
            plt.axis("off")
            plt.show()
        
        original_img = cv2.cvtColor(original_img, cv2.COLOR_RGB2BGR)
        out_filename = save if isinstance(save, str) else 'procced_' + os.path.basename(image_path)
        cv2.imwrite('procced_images/'+ out_filename, original_img) 
        print('The prediction was successfully saved by : '+ 'procced_images/'+ out_filename)
        
    
    def _preprocess_frame(self, image, target_size=640):
        """
        Resize an image while maintaining its aspect ratio and adding black padding to match the target size.

        Args:
            image (numpy.ndarray): Input image in BGR format.
            target_size (int, optional): Desired output size for YOLO detection (default is 640).

        Returns:
            numpy.ndarray: Preprocessed image with maintained aspect ratio and black padding.
        """
        h, w = image.shape[:2]
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
    
        resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
        padded_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        left = (target_size - new_w) // 2
        top = (target_size - new_h) // 2
        padded_img[top:top + new_h, left:left + new_w] = resized_img
    
        return padded_img

    def _resize_and_pad_frame(self, h, w, target_size=640):
        """
        Compute scaling factor and padding values needed to resize an image while keeping its aspect ratio.

        Args:
            h (int): Original image height.
            w (int): Original image width.
            target_size (int, optional): Desired output size (default is 640).

        Returns:
            tuple: (scale factor, left padding, top padding, (original width, original height)).
        """
        scale = min(target_size / w, target_size / h)

        new_w = int(w * scale)
        new_h = int(h * scale)

        delta_w = target_size - new_w
        delta_h = target_size - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        return scale, left, top, (w, h)

    def predict_video(self, video_path, target_size=640, show=False, save=False):
        """
        Perform object detection on a video file and save the processed output.

        Args:
            video_path (str): Path to the input video file.
            target_size (int, optional): The desired size for processing frames (default is 640).
            show (bool, optional): Whether to display the processed video (default is True).
            save (bool or str, optional): If True, saves the output video with detections.
                If a string is provided, it is used as the output filename.
        """
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('local_save.mp4', fourcc, fps, (target_size, target_size))
        
        while True:
            ret, frame = cap.read()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if frame is None:
                break
        
            processed_frame = self._preprocess_frame(frame)
            out.write(processed_frame)
        
        cap.release()
        out.release()

        prediction = self.model.predict('local_save.mp4', stream=True)
        predict_boxes = [frame.boxes for frame in prediction]

        cap = cv2.VideoCapture(video_path)
        width, height = int(cap.get(3)), int(cap.get(4))
        scale, left, top, original_size = self._resize_and_pad_frame(height, width)
        
        out_filename = save if isinstance(save, str) else 'procced_' + os.path.basename(video_path)
        out = cv2.VideoWriter('procced_videos/'+out_filename, fourcc, fps, (width, height))
        
        frame_num = 0
        while True:
            ret, frame = cap.read()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if frame is None:
                break

            detections = predict_boxes[frame_num]
            frame_num += 1
            processed_detections = self._rescale_detections(detections, scale, left, top, original_size)
            processed_frame = self._draw_detections(frame, processed_detections)
            out.write(processed_frame)
                
        cap.release()
        out.release()
        print('The prediction was successfully saved by : '+ 'procced_videos/'+out_filename)
        os.remove('local_save.mp4')