from keras import applications

# THIS FILE IS JUST TO DOWNLOAD THE WEIGTHS FOR THE FIRST TIMA
vgg_model = applications.VGG16(weights='imagenet', include_top=True)
vgg_model.summary()