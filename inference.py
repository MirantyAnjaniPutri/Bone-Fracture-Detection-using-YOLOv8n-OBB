from ultralytics import YOLO
import os
import random
import shutil
from PIL import Image
import matplotlib.pyplot as plt

def display_results(results_ori, results_histo):
    fig, axs = plt.subplots(2, len(results_ori), figsize=(15, 8))

    # Display results for the original dataset
    for i, result in enumerate(results_ori):
        axs[0, i].imshow(result.img)
        axs[0, i].set_title(f"Original - Image {i+1}")
        for bbox, label, conf in zip(result.xyxy, result.names, result.scores):
            axs[0, i].add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, color='red'))
            axs[0, i].text(bbox[0], bbox[1], f'{label} {conf:.2f}', color='red')

    # Display results for the histogram equalized dataset
    for i, result in enumerate(results_histo):
        axs[1, i].imshow(result.img)
        axs[1, i].set_title(f"Histogram Equalized - Image {i+1}")
        for bbox, label, conf in zip(result.xyxy, result.names, result.scores):
            axs[1, i].add_patch(plt.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, color='blue'))
            axs[1, i].text(bbox[0], bbox[1], f'{label} {conf:.2f}', color='blue')

    plt.tight_layout()
    plt.show()

def main():
    # Define model configurations
    model_weights_ori = 'C:/Users/miran/runs/obb/yolov8n-obb/weights/best.pt'
    model_weights_histo = 'C:/Users/miran/runs/obb/yolov8n-obb2/weights/best.pt'

    # Initialize the YOLO model with the best.pt weights
    model_ori = YOLO(model_weights_ori)
    model_histo = YOLO(model_weights_histo)

    # image_path_ori = "C:/Users/miran/code_skripsi/yolo/datasets/test/images/28_jpg.rf.e94435d98cb556f0a56cff1d7dd9b920.jpg"
    # image_path_histo = "C:/Users/miran/code_skripsi/yolo/hist_datasets/test/images/28_jpg.rf.e94435d98cb556f0a56cff1d7dd9b920.jpg"

    # # New image paths
    # image_path_ori_copy = "C:/Users/miran/code_skripsi/yolo/datasets/test/images/28_jpg.rf.e94435d98cb556f0a56cff1d7dd9b920-copy.jpg"
    # image_path_histo_copy = "C:/Users/miran/code_skripsi/yolo/hist_datasets/test/images/28_jpg.rf.e94435d98cb556f0a56cff1d7dd9b920-copy.jpg"

    # # Copy the original images to new filenames
    # shutil.copy(image_path_ori, image_path_ori_copy)
    # shutil.copy(image_path_histo, image_path_histo_copy)

    # # Perform object detection on the image
    # results_1 = model_ori.predict(image_path_ori, save=True, conf=0.3)
    # results_2 =  model_ori.predict(image_path_histo, save=True, conf=0.3)
    # results_3 = model_histo.predict(image_path_ori_copy, save=True, conf=0.3)
    # results_4 = model_histo.predict(image_path_histo_copy, save=True, conf=0.3)

    # # Run batched inference on a list of images
    # results = model_ori(["C:/Users/miran/code_skripsi/yolo/datasets/test/images/28_jpg.rf.e94435d98cb556f0a56cff1d7dd9b920.jpg",
    #                      "C:/Users/miran/code_skripsi/yolo/hist_datasets/test/images/28_jpg.rf.e94435d98cb556f0a56cff1d7dd9b920.jpg"])  # return a list of Results objects
    # results_histo = model_histo(["C:/Users/miran/code_skripsi/yolo/datasets/test/images/28_jpg.rf.e94435d98cb556f0a56cff1d7dd9b920.jpg",
    #                      "C:/Users/miran/code_skripsi/yolo/hist_datasets/test/images/28_jpg.rf.e94435d98cb556f0a56cff1d7dd9b920.jpg"])

     # Define image paths for both datasets
    image_paths_ori = [
        "C:/Users/miran/code_skripsi/yolo/datasets/test/images/28_jpg.rf.e94435d98cb556f0a56cff1d7dd9b920.jpg",
        # Add more image paths from the original dataset if needed
    ]
    image_paths_histo = [
        "C:/Users/miran/code_skripsi/yolo/hist_datasets/test/images/28_jpg.rf.e94435d98cb556f0a56cff1d7dd9b920.jpg",
        # Add more image paths from the histogram equalized dataset if needed
    ]

    # Concatenate the image paths
    all_image_paths = image_paths_ori + image_paths_histo

    # Perform batched inference on all images
    results_ori = model_ori(all_image_paths)  # Batched inference with model_ori
    results_histo = model_histo(all_image_paths)  # Batched inference with model_histo

    # Display the results for both datasets
    display_results(results_ori, results_histo)

    # # Process results list
    # for result in results_ori:
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs
    #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     probs = result.probs  # Probs object for classification outputs
    #     obb = result.obb  # Oriented boxes object for OBB outputs
    #     result.show()  # display to screen

    # # Process results list
    # for result in results_histo:
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     masks = result.masks  # Masks object for segmentation masks outputs
    #     keypoints = result.keypoints  # Keypoints object for pose outputs
    #     probs = result.probs  # Probs object for classification outputs
    #     obb = result.obb  # Oriented boxes object for OBB outputs
    #     result.show()  # display to screen



if __name__ == '__main__':
    main()
