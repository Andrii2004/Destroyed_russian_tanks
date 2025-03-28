# Destroyed_russian_tanks

This project aims to use computer vision for automatic detection and assessment of damage to Russian tanks. Using image processing algorithms and Ultralytics' trained YOLO 11 model, the system can analyse images and videos of tanks destroyed or damaged during combat operations, determine the level of damage and provide data for further analysis or decision-making. The project aims to speed up the damage assessment process and help in the operational planning of military operations.

## ⚠️ Disclaimer 

The project is still under development, so now only the detection of tanks and their classification into groups ‘With tower’ and ‘Without tower’ are available

## YOLO 11 🤖

YOLO11 is the latest iteration in the Ultralytics YOLO series of real-time object detectors, redefining what's possible with cutting-edge accuracy, speed, and efficiency. Building upon the impressive advancements of previous YOLO versions, YOLO11 introduces significant improvements in architecture and training methods, making it a versatile choice for a wide range of computer vision tasks.

![performance-comparison](https://github.com/user-attachments/assets/1e355ca4-26a7-438e-8ce8-375da9dc5557)



## Dataset 🗂
### Data sourses 
the initial dataset consists of images based on data from the dataset (https://www.kaggle.com/datasets/piterfm/2022-ukraine-russia-war-equipment-losses-oryx) and data from public sources

### Data preproccesing pipeline
- Cutting collages (images consisting of several photos)
- Data filtering
(removing images with low resolution
or those where it is impossible to determine
whether the tank is accurately depicted)
- Image labeling
- Auto-Orient
- Resize (Fit edges to 640x640 while keeping the aspect ratio by adding black rectangles)

### Images augmentation steps 
- 90° Rotate 
- Rotation
- Grayscale

## Example of usage 🛠️
New object creation 
```
object_name = Tank_recogniser('path/to/pretrained/model')
```

Image-based prediction 
```
object_name.predict_image('path/to/image', show=True)
```

Video-based prediction
```
object_name.predict_video('path/to/video') 
```
More detailes you can find in Example notebook
## Screenshots 📷
Here are some screenshots that demonstrate the functionality of Tank_recogniser :

![procced_image1](https://github.com/user-attachments/assets/fe4e0dcf-c46e-4901-bce0-86d6ecce9b9a)
![procced_image2](https://github.com/user-attachments/assets/84af7b34-7fb1-4413-ab67-fb0ddb90017d)
![procced_image3](https://github.com/user-attachments/assets/3557dabf-9cf3-4c31-b88d-426690ad163a)
![procced_image4](https://github.com/user-attachments/assets/2658a12a-bce1-48dc-a943-7256be8f6168)

## Future plans 🚀

- Add a separate model for a more detailed predict of the tank's condition
- Add other groups of equipment, such as Infantry fighting vehicle , Armoured personnel carrier , etc.
