'''
Code for fine-tuning VGG16 for a new task.
'''

from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout, Convolution2D
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.optimizers import Nadam, SGD
import tensorflow as tf
tf.python.control_flow_ops = tf
import matplotlib.pyplot as plt
from keras import backend as K
K.set_image_dim_ordering('tf')

# Keep track of training/validation loss
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

history = LossHistory()
nb_train_samples = 23000
nb_validation_samples = 2000
nb_epoch = 1000
batch_size = 4

# VGG uses 224x224 pixel image size
IMSIZE = (224, 224)

# Training directory
train_dir = '/home/matt/kaggle/c_vs_d/train_1'
# Testing directory
val_dir = '/home/matt/kaggle/c_vs_d/validation_1'

# Start with an VGG16 model, not including the final softmax layer.
base_model = VGG16(weights='imagenet', include_top=True)
print 'Loaded VGG16 model'

# Turn off training on first few convolution layers
for layer in base_model.layers[:7]:
    layer.trainable = False

predictions = Dense(2, activation='sigmoid', name='predictions')(base_model.get_layer('fc2').output)
model = Model(input=base_model.input, output=predictions)

# Show some debug output
model.summary()

# Load weights from previously trained model
#model.load_weights('/home/matt/kaggle/c_vs_d/VGG16.hdf5')


# Data generators for feeding training/validation images to the model.
# Link to all adjustable parameters: https://keras.io/preprocessing/image/
train_datagen = ImageDataGenerator(rotation_range=45, # Degree range for random rotations
        fill_mode='reflect', # One of {"constant", "nearest", "reflect" or "wrap"}.
        horizontal_flip=True, # Randomly flip inputs horizontally.
        vertical_flip=True, # Randomly flip inputs vertically.
        rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMSIZE,  # all images will be resized to 224x224
        batch_size=batch_size,
        class_mode='categorical')

val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
        val_dir, 
        target_size=IMSIZE,  # all images will be resized to 224x224
        batch_size=batch_size,
        class_mode='categorical')

# Training model and saving weights
def fit_model(save_round, lr):
    save_weight_file = "/home/matt/kaggle/c_vs_d/incep_round" + str(save_round) + "v4.hdf5"
    checkpointer = ModelCheckpoint(filepath=save_weight_file, verbose=1, save_best_only=True)
    earlystopping = EarlyStopping(monitor = 'val_loss', patience = 10, verbose = 1, mode = 'auto')
    model.compile(loss='binary_crossentropy', optimizer=Nadam(lr=lr), metrics=['accuracy'])
    model.fit_generator(train_generator, nb_epoch=nb_epoch, samples_per_epoch=nb_train_samples,
                                 validation_data = val_generator,
                                 callbacks = [checkpointer, earlystopping, history],
                                 nb_val_samples=nb_validation_samples)
    model.load_weights(save_weight_file)

fit_model(0, 1e-6)
#fit_model(1, 3e-7)
#fit_model(2, 1e-7)


# Graph history of train/validation loss over epochs
#loss = history.losses
#val_loss = history.val_losses

#plt.xlabel('Epochs')
#plt.ylabel('Loss')
#plt.title('Loss Trend')
#plt.plot(loss, 'blue', label='Training Loss')
#plt.plot(val_loss, 'green', label='Validation Loss')
#plt.xticks(range(0,len(loss))[0::2])
#plt.legend()
#plt.show()
