import csv
import numpy as np
import cv2
def get_training_data():
    lines=[]
    with open('../data/driving_log.csv') as csv_file:
        reader = csv.DictReader(csv_file)
        for line in reader:
            lines.append(line)
    images = []
    measurements = []
    current_path = '../data/IMG/'
    correction_factor = [0.0,0.2,-0.2]
    for line in lines:
        for i,position in enumerate(['center','left','right']):
            source_path = line['center']
            file_name = source_path.split('/')[-1]
            image = cv2.imread(current_path + file_name)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            images.append(image)
            measurement = float(line['steering'])
            measurements.append(measurement-correction_factor[i])
            image_flipped = np.fliplr(image)
            images.append(image_flipped)
            measurements.append(-measurement-correction_factor[i])
    return np.array(images),np.array(measurements)
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
    
def pipeline():    
    X_train,y_train = get_training_data()
    print(len(y_train))
    cnn_model(X_train,y_train)
pipeline()
    
    