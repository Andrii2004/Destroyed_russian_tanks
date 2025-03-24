# Destroyed_russian_tanks

This project aims to use computer vision for automatic detection and assessment of damage to Russian tanks. Using image processing algorithms and Ultralytics' trained YOLO v11 model, the system can analyse images and videos of tanks destroyed or damaged during combat operations, determine the level of damage and provide data for further analysis or decision-making. The project aims to speed up the damage assessment process and help in the operational planning of military operations.

## YOLO v11



## Dataset
### Data sourses 
the initial dataset consists of images based on data from the dataset (https://www.kaggle.com/datasets/piterfm/2022-ukraine-russia-war-equipment-losses-oryx) and data from public sources

### Data preproccesing pipeline
- Cutting collages (images consisting of several photos)
- Data filtering
(removing images with low resolution
or those where it is impossible to determine
whether the tank is accurately depicted)
- Image markup¶
- Auto-Orient
- Resize (Fit edges to 640x640 while keeping the aspect ratio by adding black rectangles)

### Images augmentation steps 
- 90° Rotate 
- Rotation
- Grayscale

## Screenshots 📷


