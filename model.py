# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-12T10:34:55.115331Z","iopub.execute_input":"2023-07-12T10:34:55.115705Z","iopub.status.idle":"2023-07-12T10:34:55.128235Z","shell.execute_reply.started":"2023-07-12T10:34:55.115672Z","shell.execute_reply":"2023-07-12T10:34:55.126786Z"}}
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os
import math
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
# import warnings
# warnings.filterwarnings("ignore")

print('Running')

xpath='/home/20bce131/Turkey/img_dir'
ypath='/home/20bce131/Turkey/ann_dir'
trn_hist_path='/home/20bce131/model/val_imgs_out'
val_out_path='/home/20bce131/model/train_hist'

try:
    os.mkdir(trn_hist_path)
    os.mkdir(val_out_path)
    print('Created necessary directories')
except:
    print('Necessary Directories already exist')

#Hyperparameters and Config
#Training
batch_size = 64
num_epochs = 100
learning_rate=1e-2
momentum = 0.9
optimizer_alg = 'adam' #'adam' or 'sgd'
#LR Scheduler
gamma = 0.9
decay_steps = 10
#Early stopping
min_delta=1e-3
patience=3
#Outputs
num_val_imgs_out=5
#Network
dropout_rate = 0.2

print(f'Batch Size : {batch_size}')
print(f'Num Epochs : {num_epochs}')
print(f'learning_rate : {learning_rate}')
print(f'Optimizer : {optimizer_alg}')
print(f'Dropout rate : {dropout_rate}')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-12T10:34:55.130071Z","iopub.execute_input":"2023-07-12T10:34:55.130974Z","iopub.status.idle":"2023-07-12T10:34:55.147450Z","shell.execute_reply.started":"2023-07-12T10:34:55.130912Z","shell.execute_reply":"2023-07-12T10:34:55.146480Z"}}
def visualize_sample(img, label, title, path=''):
    label = tf.transpose(label, perm=[2,0,1])
    fig, ax = plt.subplots(1, 6, figsize=(20,20))
    for i, subplot_ax in zip(range(5 + 1), ax.flatten()):
        if i == 0: 
            subplot_ax.imshow(img)
            subplot_ax.set_title(title)
        else:
            subplot_ax.imshow(label[i-1], cmap='gray', vmin=0, vmax=1)
            subplot_ax.set_title(f'Label {i}')
    if path != '':
        plt.savefig(f'{path}/{title}.png')
    
def get_iou(preds, label):
    classwise_iou = []
    num_classes = 5
    preds = tf.argmax(preds, axis=-1)
    label = tf.argmax(label, axis=-1)
    for c in range(num_classes):
        tp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(label, c), tf.equal(preds, c)), dtype=tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.not_equal(label, c), tf.equal(preds, c)), dtype=tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(label, c), tf.not_equal(preds, c)), dtype=tf.float32))
        iou = tp / (tp + fn + fp)
        if tf.math.is_nan(iou): continue
        classwise_iou.append(iou)
    classwise_iou = tf.stack(classwise_iou)
    miou = tf.reduce_mean(classwise_iou)
    return miou

def get_sample_from_val(idx):
    idx_num = idx % batch_size
    batch_num = int((idx-idx_num) / batch_size)
    imgs, labels = val.__getitem__(batch_num)
    img, label = imgs[idx_num], labels[idx_num]
    return img, label

def get_random_val_sample():
    batch = np.random.randint(0, len(val),(1,))
    idx = int(np.random.randint(0, batch_size,(1,)))
    imgs, labels = val.__getitem__(int(batch))
    return imgs[idx], labels[idx]

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-12T10:34:55.267105Z","iopub.execute_input":"2023-07-12T10:34:55.268237Z","iopub.status.idle":"2023-07-12T10:34:55.293740Z","shell.execute_reply.started":"2023-07-12T10:34:55.268191Z","shell.execute_reply":"2023-07-12T10:34:55.292670Z"}}
class Train(tf.keras.utils.Sequence):
    def __init__(self, imgs, batch_size=batch_size):
        self.num_classes = 5
        self.height, self.width = 256, 256
        self.xtrain_path = xpath
        self.ytrain_path = ypath
        self.imgs = imgs
        self.batch_size = batch_size
        
        self.xtrain_filenames = [os.path.join(self.xtrain_path, img_name) for img_name in self.imgs]
        self.ytrain_filenames = [os.path.join(self.ytrain_path, img_name) for img_name in self.imgs]

    def __getitem__(self, idx):
        batch_x_filenames = self.xtrain_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y_filenames = self.ytrain_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch_x = []
        batch_y = []
        for x_filename, y_filename in zip(batch_x_filenames, batch_y_filenames):
            xtrain = tf.keras.preprocessing.image.load_img(x_filename, target_size=(self.height, self.width))
            xtrain = tf.keras.preprocessing.image.img_to_array(xtrain)
            batch_x.append(xtrain)

            ytrain = tf.keras.preprocessing.image.load_img(y_filename, target_size=(self.height, self.width), color_mode='grayscale')
            ytrain = tf.keras.preprocessing.image.img_to_array(ytrain)
            ytrain = tf.squeeze(ytrain)
            ytrain = self.process_label(ytrain)
            ytrain = tf.transpose(ytrain, perm=[1, 2, 0])
            batch_y.append(ytrain)

        batch_x = tf.stack(batch_x)
        batch_y = tf.stack(batch_y)
        batch_x = batch_x / 255.0

        return batch_x, batch_y

    def __len__(self):
        return math.ceil(len(self.imgs) / self.batch_size)
    
    def process_label(self, label):
        r = []
        for i in range(self.num_classes):
            mask = tf.cast(tf.equal(label, i+1), dtype=tf.float32)
            r.append(mask)
        return tf.stack(r)
    
class Validate(tf.keras.utils.Sequence):
    def __init__(self, imgs, batch_size=batch_size):
        self.num_classes = 5
        self.height, self.width = 256, 256
        self.xtrain_path = xpath
        self.ytrain_path = ypath
        self.imgs = imgs
        self.batch_size = batch_size
        
        self.xtrain_filenames = [os.path.join(self.xtrain_path, img_name) for img_name in self.imgs]
        self.ytrain_filenames = [os.path.join(self.ytrain_path, img_name) for img_name in self.imgs]

    def __getitem__(self, idx):
        batch_x_filenames = self.xtrain_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_y_filenames = self.ytrain_filenames[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch_x = []
        batch_y = []
        for x_filename, y_filename in zip(batch_x_filenames, batch_y_filenames):
            xtrain = tf.keras.preprocessing.image.load_img(x_filename, target_size=(self.height, self.width))
            xtrain = tf.keras.preprocessing.image.img_to_array(xtrain)
            batch_x.append(xtrain)

            ytrain = tf.keras.preprocessing.image.load_img(y_filename, target_size=(self.height, self.width), color_mode='grayscale')
            ytrain = tf.keras.preprocessing.image.img_to_array(ytrain)
            ytrain = tf.squeeze(ytrain)
            ytrain = self.process_label(ytrain)
            ytrain = tf.transpose(ytrain, perm=[1, 2, 0])
            batch_y.append(ytrain)

        batch_x = tf.stack(batch_x)
        batch_y = tf.stack(batch_y)
        batch_x = batch_x / 255.0

        return batch_x, batch_y

    def __len__(self):
        return math.ceil(len(self.imgs) / self.batch_size)
    
    def process_label(self, label):
        r = []
        for i in range(self.num_classes):
            mask = tf.cast(tf.equal(label, i+1), dtype=tf.float32)
            r.append(mask)
        return tf.stack(r)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-12T10:34:55.297300Z","iopub.execute_input":"2023-07-12T10:34:55.298061Z","iopub.status.idle":"2023-07-12T10:34:55.354554Z","shell.execute_reply.started":"2023-07-12T10:34:55.298004Z","shell.execute_reply":"2023-07-12T10:34:55.353683Z"}}
imgs = np.array(os.listdir(xpath))
train_imgs, val_imgs = train_test_split(imgs, test_size=0.2)
train = Train(train_imgs)
val = Validate(val_imgs)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-12T10:34:55.356988Z","iopub.execute_input":"2023-07-12T10:34:55.357572Z","iopub.status.idle":"2023-07-12T10:34:55.378442Z","shell.execute_reply.started":"2023-07-12T10:34:55.357536Z","shell.execute_reply":"2023-07-12T10:34:55.377580Z"}}
def double_conv(in_channels, out_channels):
    conv = tf.keras.Sequential([
        layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding='same'),
        layers.ReLU(),
#         layers.SpatialDropout2D(rate=dropout_rate),
        layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding='same'),
        layers.ReLU(),
        layers.BatchNormalization(),
        layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding='same'),
        layers.ReLU(),
#         layers.SpatialDropout2D(rate=dropout_rate),
        layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding='same'),
        layers.ReLU(),
        layers.BatchNormalization(),
#       layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding='same'),
#       layers.ReLU(),
#       layers.SpatialDropout2D(rate=dropout_rate),
#       layers.Conv2D(filters=out_channels, kernel_size=3, strides=1, padding='same'),
#       layers.ReLU(),
#       layers.BatchNormalization()
    ])
    return conv

def double_Tconv(in_channels, out_channels):
    Tconv = tf.keras.Sequential([
        layers.Conv2DTranspose(filters=out_channels, kernel_size=3, strides=1, padding='same'),
        layers.ReLU(),
#         layers.SpatialDropout2D(rate=dropout_rate), 
        layers.Conv2DTranspose(filters=out_channels, kernel_size=3, strides=1, padding='same'),
        layers.ReLU(),
        layers.BatchNormalization(),
        layers.Conv2DTranspose(filters=out_channels, kernel_size=3, strides=1, padding='same'),
        layers.ReLU(),
#         layers.SpatialDropout2D(rate=dropout_rate),
        layers.Conv2DTranspose(filters=out_channels, kernel_size=3, strides=1, padding='same'),
        layers.ReLU(),
        layers.BatchNormalization(),
#       layers.Conv2DTranspose(filters=out_channels, kernel_size=3, strides=1, padding='same'),
#       layers.ReLU(),
#         layers.SpatialDropout2D(rate=dropout_rate),
#       layers.Conv2DTranspose(filters=out_channels, kernel_size=3, strides=1, padding='same'),
#       layers.ReLU(),
#       layers.BatchNormalization()
    ])
    return Tconv

class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        self.max_pool = layers.MaxPool2D(pool_size=2, strides=2, padding='same')
        self.upsample = layers.UpSampling2D()
        self.conv1 = double_conv(3, 64)
        self.conv2 = double_conv(64, 128)
        self.conv3 = double_conv(128, 256)
        self.conv4 = double_conv(256, 512)
        self.conv5 = double_conv(512, 1024)
        self.Tconv5 = double_Tconv(1024, 512)
        self.Tconv4 = double_Tconv(1024, 256)
        self.Tconv3 = double_Tconv(512, 128)
        self.Tconv2 = double_Tconv(256, 64)
        self.Tconv1 = double_Tconv(128, 32)
        self.bottleneck = layers.Conv2D(filters=5, kernel_size=1, strides=1)
        self.softmax = layers.Softmax(axis=3)
        self.dropout = layers.SpatialDropout2D(rate=dropout_rate)
    
    def call(self, inputs):
        
        #inputs : batch, 256, 256, 3
        
        c1 = self.conv1(inputs) #batch, 256, 256, 64
        x1 = self.max_pool(c1) #batch, 128,128, 64

        c2 = self.conv2(x1) #batch, 128, 128, 128
        x2 = self.max_pool(c2) #batch, 64, 64, 128
        
        c3 = self.conv3(x2) #batch, 64, 64, 256
        x3 = self.max_pool(c3) #batch, 32, 32, 256

        c4 = self.conv4(x3) #batch, 32, 32, 512
        x4 = self.max_pool(c4) #batch, 16, 16, 512
        
        i = self.conv5(x4) #batch, 16, 16, 1024
        
        i = self.dropout(i) #batch, 16, 16, 1024
        
        t4 = self.Tconv5(i) #batch, 16, 16, 512
    
        z4 = tf.concat([self.upsample(t4), c4], axis=-1) #batch, 32, 32, 1024
        t3 = self.Tconv4(z4) #batch, 32, 32, 256
        
        z3 = tf.concat([self.upsample(t3), c3], axis=-1) #batch, 64, 64, 512
        t2 = self.Tconv3(z3) #batch, 64, 64, 128
    
        z2 = tf.concat([self.upsample(t2), c2], axis=-1) #batch, 32, 32, 256
        t1 = self.Tconv2(z2) #batch, 128, 128, 64
        
        z1 = tf.concat([self.upsample(t1), c1], axis=-1) #batch, 32, 256, 256, 128
    
        z0 = self.Tconv1(z1) #batch, 256, 256, 32
        logits = self.bottleneck(z0) #batch, 256, 256, 5
        out = self.softmax(logits) #batch, 256, 256, 5
        return out

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-12T10:34:55.381092Z","iopub.execute_input":"2023-07-12T10:34:55.381747Z","iopub.status.idle":"2023-07-12T10:35:05.952295Z","shell.execute_reply.started":"2023-07-12T10:34:55.381714Z","shell.execute_reply":"2023-07-12T10:35:05.951477Z"}}
model = Network()
model.build(input_shape=(batch_size,256,256,3))
model.summary()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2023-07-12T10:35:05.954972Z","iopub.execute_input":"2023-07-12T10:35:05.955302Z"}}
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = learning_rate,
    decay_steps=decay_steps,
    decay_rate=gamma,
    staircase=True
)
callbacks=[
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience, verbose=1)
]

if optimizer_alg == 'sgd':
    optim = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=momentum, nesterov=True)
elif optimizer_alg == 'adam':
    optim = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
else: optim = 'sgd'
    

model.compile(optimizer=optim, loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
hist = model.fit(train, steps_per_epoch=train.__len__(), epochs=num_epochs, validation_data=val, verbose=2)

# %% [code] {"jupyter":{"outputs_hidden":false}}
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.plot(hist.history['loss'], label='Training Loss')
plt.plot(hist.history['val_loss'], label='Validation Loss')
plt.legend()
plt.savefig(f'{trn_hist_path}/loss_curve.png')
plt.show()
plt.clf()

# %% [code] {"jupyter":{"outputs_hidden":false}}
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.plot(hist.history['accuracy'], label='Training Accuracy')
plt.plot(hist.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.savefig(f'{trn_hist_path}/acc_curve.png')

# %% [code] {"jupyter":{"outputs_hidden":false}}
preds = model.predict(val)
iou = get_iou(preds, val.ytrain)
print(f'Mean IoU : {iou}')

# %% [code] {"jupyter":{"outputs_hidden":false}}
for i in range(num_val_imgs_out):
    img, label = get_random_val_sample()
    visualize_sample(img, label, f'Ground Truth {val_imgs[i]}', path=val_out_path)
    pred = model.predict(tf.expand_dims(img, axis=0))
    pred = tf.squeeze(pred)
    pred = tf.argmax(pred, axis=-1) + 1
    pred = val.process_label(pred)
    pred = tf.transpose(pred, perm=[1,2,0])
    visualize_sample(img, tf.squeeze(pred), f'Model Predictions {val_imgs[i]}', path=val_out_path)