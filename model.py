import csv
import numpy as np
import cv2
'''
    The function get_training_data does the following, 
    1. From the driving_log.csv, it reads the center image path and steering angle
    2. Flip the center image - Data Augmentation and reverse the angle measurement
    3. Returns the images and angle measurements numpy arrays
'''
def get_training_data():
    lines=[]
    with open('../data/driving_log.csv') as csv_file:
        reader = csv.DictReader(csv_file)
        for line in reader:
            lines.append(line)
    images = []
    measurements = []
    current_path = '../data/IMG/'
    correction_factor = [0.0]#,0.2,-0.2]
    flipped_correction_factor = [0.0]#,-0.2,0.2]
    for line in lines:
        for i,position in enumerate(['center']):#,'left','right']):
            source_path = line[position]
            file_name = source_path.split('/')[-1]
            image = cv2.imread(current_path + file_name)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            images.append(image)
            measurement = float(line['steering'])
            measurements.append(measurement-correction_factor[i])
            image_flipped = np.fliplr(image)
            images.append(image_flipped)
            measurements.append(-measurement-flipped_correction_factor[i])
    return np.array(images),np.array(measurements)
'''
    The CNN model inspired from NVIDIA end-2-end deeplearning SDC CNN model
    It has 1 normalization layer,1 preprocessing layer(cropping),5 convolution layers 
    and 3 fully connected layers. This model does not use the generator to train.
'''
def cnn_model(X_train,y_train):
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D
    from keras.layers.pooling import MaxPooling2D
    model = Sequential()
    model.add(Lambda(lambda x : (x / 255.0) - 0.5,input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25),(0,0))))
    
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    #model.add(MaxPooling2D())
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    #model.add(MaxPooling2D())
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train,y_train,validation_split=0.2,shuffle=True,epochs=5)
    model.save('model.h5')
'''
    The generator function which yields the training and validation datasets
    with the batch_size given as and when called to avoid loading all the data
    in main memory for training.
'''
import sklearn
def generator(samples,batch_size=64):
    num_samples = len(samples)
    while 1:
        sklearn.utils.shuffle(samples)
        current_path = '../COLLECTED_DATA/IMG/'
        correction_factor = [0.0]#,0.2,-0.2]
        flipped_correction_factor = [0.0]#,-0.2,0.2]
        for offset in range(0,num_samples,batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i,position in enumerate(['center']):#,'left','right']):
                    source_path = batch_sample[position]
                    file_name = source_path.split('\\')[-1]
                    image = cv2.imread(current_path + file_name)
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    images.append(image)
                    angle = float(batch_sample['steering'])
                    angles.append(angle-correction_factor[i])
                    image_flipped = np.fliplr(image)
                    images.append(image_flipped)
                    angles.append(-angle-flipped_correction_factor[i])
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train,y_train)

'''
    The CNN model inspired from NVIDIA end-2-end deeplearning SDC CNN model
    It has 1 normalization layer,1 preprocessing layer(cropping),5 convolution layers and 3 fully 
    connected layers and 2 dropout layers. This model does use the generator to train.
'''
def generator_cnn_model(train_samples,validation_samples):
    batch_size = 32
    train_generator = generator(train_samples,batch_size=batch_size)
    validation_generator = generator(validation_samples,batch_size=batch_size)
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout
    from keras.layers.pooling import MaxPooling2D
    model = Sequential()
    model.add(Lambda(lambda x : (x / 255.0) - 0.5,input_shape=(160,320,3))) # Normalization layer - Mean Centered
    model.add(Cropping2D(cropping=((70,25),(0,0)))) # Cropping only the ROI area
    
    model.add(Convolution2D(24,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(36,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(48,5,5,subsample=(2,2),activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Convolution2D(64,3,3,activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.25))
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator,samples_per_epoch=int(len(train_samples)/batch_size),validation_data=validation_generator,nb_val_samples=int(len(validation_samples)/batch_size),epochs=3)
    model.save('generator_model.h5')
'''
   The get_train_validation_samples splits the data into train and validation samples 
   and returns
'''
def get_train_validation_samples():
    samples = []    
    with open('../COLLECTED_DATA/driving_log.csv') as csv_file:
        reader = csv.DictReader(csv_file)
        for line in reader:
            samples.append(line)
    from sklearn.model_selection import train_test_split
    train_samples,validation_samples = train_test_split(samples,test_size=0.2)
    return train_samples,validation_samples
'''
   This pipeline uses the generator to train the CNN model.
'''
def generator_pipeline():
    train_samples,validation_samples = get_train_validation_samples()
    generator_cnn_model(train_samples,validation_samples)
'''
    This pipeline does not use the generator to train the CNN model. 
'''
def pipeline():    
    X_train,y_train = get_training_data()
    print(len(y_train))
    cnn_model(X_train,y_train)
generator_pipeline()
#pipeline()
    
    