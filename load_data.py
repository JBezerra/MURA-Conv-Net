import os
import time
import cv2
import numpy as np

def removeDS(path):
    if '.DS_Store' in path:
        path.remove('.DS_Store')

x_train = []
y_train = []

x_val = []
y_val = []

def load_training_data(save_data = False):

    print "Loading Training Data..."
    start_time = time.time()

    BASE_DIR = './dataset-MURA-v1.1/train'
    bodyparts_dir = os.listdir(BASE_DIR)
    removeDS(bodyparts_dir)
    for index,part in enumerate(bodyparts_dir):
        part_dir = "{}/{}".format(BASE_DIR,part)
        patients = os.listdir(part_dir)

        removeDS(patients)
        print "Working into {} {}/{}...".format(part,index+1,len(bodyparts_dir))

        for patient in patients:
            patient_dir = "{}/{}".format(part_dir,patient)
            studies = os.listdir(patient_dir)

            removeDS(studies)
        
            for study in studies:
                study_dir = "{}/{}".format(patient_dir,study)
                images = os.listdir(study_dir)

                removeDS(images)
                
                split = study.split('_')
                label = split[1]

                for image in images:
                    # Check the Images
                    img_dir = "{}/{}".format(study_dir,image)
                    image_gray = cv2.imread(img_dir).astype(np.float32)
                    image_gray /= 255
                    x_train.append(image_gray)
                    y_train.append(0 if label == "negative" else 1)
    
    elapsed_time = time.time() - start_time
    print "Time Elapsed To Load Data: {} min.".format(elapsed_time/60)

    if save_data:
        np.save("x_train.npy",x_train)
        np.save("y_train.npy",y_train)

    return x_train,y_train

def load_validation_data(save_data = False):

    print "Loading Validation Data..."
    start_time = time.time()

    BASE_DIR = './dataset-MURA-v1.1/valid'
    bodyparts_dir = os.listdir(BASE_DIR)
    removeDS(bodyparts_dir)
    for index,part in enumerate(bodyparts_dir):
        part_dir = "{}/{}".format(BASE_DIR,part)
        patients = os.listdir(part_dir)

        removeDS(patients)
        print "Working into {} {}/{}...".format(part,index+1,len(bodyparts_dir))

        for patient in patients:
            patient_dir = "{}/{}".format(part_dir,patient)
            studies = os.listdir(patient_dir)

            removeDS(studies)
        
            for study in studies:
                study_dir = "{}/{}".format(patient_dir,study)
                images = os.listdir(study_dir)

                removeDS(images)
                
                split = study.split('_')
                label = split[1]

                for image in images:
                    # Check the Images
                    img_dir = "{}/{}".format(study_dir,image)
                    image_gray = (cv2.imread(img_dir)).astype(np.float32)
                    image_gray /= 255
                    x_val.append(image_gray)
                    y_val.append(0 if label == "negative" else 1)
    

    if save_data:
        np.save("x_val.npy",x_val)
        np.save("y_val.npy",y_val)
    
    elapsed_time = time.time() - start_time
    print "Time Elapsed To Load Data: {} min.".format(elapsed_time/60)

    return x_val,y_val