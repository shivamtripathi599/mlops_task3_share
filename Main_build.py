#!/usr/bin/env python
# coding: utf-8


from keras.applications import MobileNet

# MobileNet was designed to work on 224 x 224 pixel input images sizes
img_rows, img_cols = 224, 224 

# Re-loads the MobileNet model without the top or FC layers
MobileNet = MobileNet(weights = 'imagenet', 
                 include_top = False, 
                 input_shape = (img_rows, img_cols, 3))

# Here we freeze the last 4 layers 
# Layers are set to trainable as True by default
for layer in MobileNet.layers:
    layer.trainable = False
    
# Let's print our layers 
for (i,layer) in enumerate(MobileNet.layers):
    print(str(i) + " "+ layer.__class__.__name__, layer.trainable)
    
units = int(256)

# We only train 5 EPOCHS 
epochs = 3

# set our batch size (typically on most mid tier systems we'll use 16-32)
batch_size = 16



def new_layer(bottom_model, num_classes, units):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""

    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(units*4,activation='relu')(top_model)
    top_model = Dense(units*2,activation='relu')(top_model)
    top_model = Dense(units,activation='relu')(top_model)
    top_model = Dense(num_classes,activation='softmax')(top_model)
    return top_model



from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

# Set our class number to 3 (Young, Middle, Old)
num_classes = 10

FC_Head = new_layer(MobileNet, num_classes, units)

model = Model(inputs = MobileNet.input, outputs = FC_Head)

print(model.summary())


from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop

train_data_dir = '/root/mlops_task_3/monkey_breed/train/'
validation_data_dir = '/root/mlops_task_3/monkey_breed/validation/'

# Let's use some data augmentaiton 
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=45,
      width_shift_range=0.3,
      height_shift_range=0.3,
      horizontal_flip=True,
      fill_mode='nearest')
 
validation_datagen = ImageDataGenerator(rescale=1./255)
 

 
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')
 
validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')

# We use a very small learning rate 
model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])





history = model.fit_generator(
    train_generator,
    steps_per_epoch = batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = batch_size)



import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

fig2 = plt.figure()
plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
fig2.savefig('/root/mlops_task_3/plots/plotacc.png')


fig1 = plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
fig1.savefig('/root/mlops_task_3/plots/plotloss.png')



final_acc = max(acc)
final_acc = final_acc*100
print(final_acc)


from os import system

if(final_acc<97):
    system("curl --user 'username:password' 192.168.43.251:8080/job/job_mail_main_fail/build?token=main_fail")
else:
    system("curl --user 'username:password' 192.168.43.251:8080/job/job_mail_main_success/build?token=main_success")
    
