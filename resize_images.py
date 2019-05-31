import os
from PIL import Image 
import time
import cv2

def removeDS(path):
    if '.DS_Store' in path:
        path.remove('.DS_Store')

img_size_check = {}

avg_width = 362
avg_height = 445

NUM_IMGS = 40561

VGG_WIDTH = VGG_HEIGHT = 224

def resize_training_data():
        
        print "Resizing Training Data to {} x {}".format(VGG_WIDTH,VGG_HEIGHT)
        start_time = time.time()
        # Navigate through dataset
        BASE_DIR = './dataset-MURA-v1.1/train'
        bodyparts_dir = os.listdir(BASE_DIR)
        removeDS(bodyparts_dir)

        for index,part in enumerate(bodyparts_dir):
                part_dir = "{}/{}".format(BASE_DIR,part)
                patients = os.listdir(part_dir)

                removeDS(patients)
                print "Working into {} {}/{}...".format(part,index+1,len(bodyparts_dir))
                # Put key into dict
                img_size_check[part] = set([])
                for patient in patients:
                        patient_dir = "{}/{}".format(part_dir,patient)
                        studies = os.listdir(patient_dir)

                        removeDS(studies)
                
                        for study in studies:
                                study_dir = "{}/{}".format(patient_dir,study)
                                images = os.listdir(study_dir)

                                removeDS(images)
                                
                                for image in images:
                                        # Check the Images
                                        img_dir = "{}/{}".format(study_dir,image)

                                        # Resize to VGG Input Shape
                                        img = cv2.imread(img_dir)
                                        img_resized = cv2.resize(img, (VGG_WIDTH,VGG_HEIGHT))
                                        cv2.imwrite(img_dir,img_resized)

                                        # # Check images Dimentions
                                        # img = Image.open(img_dir)
                                        # img_size = img.size
                                        # img_size_check[part].add(img_size)
                        
                        
        elapsed_time = time.time() - start_time
        print "Done"
        print "Time Elapsed: {}".format(elapsed_time)

        # for key in img_size_check:
        #         sizes = img_size_check[key]
        #         print key
        #         print len(sizes)
        #         print "\n"


def resize_validation_data():
        
        print "Resizing Validation Data to {} x {}".format(VGG_WIDTH,VGG_HEIGHT)
        start_time = time.time()
        # Navigate through dataset
        BASE_DIR = './dataset-MURA-v1.1/valid'
        bodyparts_dir = os.listdir(BASE_DIR)
        removeDS(bodyparts_dir)

        for index,part in enumerate(bodyparts_dir):
                part_dir = "{}/{}".format(BASE_DIR,part)
                patients = os.listdir(part_dir)

                removeDS(patients)
                print "Working into {} {}/{}...".format(part,index+1,len(bodyparts_dir))
                # Put key into dict
                img_size_check[part] = set([])
                for patient in patients:
                        patient_dir = "{}/{}".format(part_dir,patient)
                        studies = os.listdir(patient_dir)

                        removeDS(studies)
                
                        for study in studies:
                                study_dir = "{}/{}".format(patient_dir,study)
                                images = os.listdir(study_dir)

                                removeDS(images)
                                
                                for image in images:
                                        # Check the Images
                                        img_dir = "{}/{}".format(study_dir,image)

                                        # Resize to VGG Input Shape
                                        img = cv2.imread(img_dir)
                                        img_resized = cv2.resize(img, (VGG_WIDTH,VGG_HEIGHT))
                                        cv2.imwrite(img_dir,img_resized)

                                        # Check images Dimentions
                                        img = Image.open(img_dir)
                                        img_size = img.size
                                        img_size_check[part].add(img_size)
                        
                        
        elapsed_time = time.time() - start_time
        print "Done"
        print "Time Elapsed: {}".format(elapsed_time)

        for key in img_size_check:
                sizes = img_size_check[key]
                print key
                print len(sizes)
                print "\n"

