import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt


model = torch.hub.load('ultralytics/yolov8')  

# Define your favorite pet's class name
favorite_pet = 'dog'  # Change this to your favorite pet class name

def detect_and_draw_box(image_path, output_path):
    # Load image
    img = Image.open(image_path)
    
    # Perform inference
    results = model(img)
    
    # Convert results to pandas dataframe
    df = results.pandas().xyxy[0]
    
    # Load image with OpenCV
    img_cv = cv2.imread(image_path)
    
    # Draw bounding boxes
    for index, row in df.iterrows():
        if row['name'] == favorite_pet:
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, favorite_pet, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Save the output image
    cv2.imwrite(output_path, img_cv)
    
    # Display the output image
    plt.imshow(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()