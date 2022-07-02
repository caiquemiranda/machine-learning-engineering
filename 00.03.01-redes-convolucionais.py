# importação das bibliotecas necessárias

from tensorflow.keras.models import Sequential                         # Sequential é um modelo sequencial
from tensorflow.keras.layers import Conv2D                             # Conv2D é uma camada de convolução
from tensorflow.keras.layers import MaxPooling2D                       # MaxPooling2D é uma camada de pooling
from tensorflow.keras.layers import Flatten                            # Flatten é uma camada de flattening
from tensorflow.keras.layers import Dense                              # Dense é uma camada densa
from tensorflow.keras.preprocessing.image import ImageDataGenerator    # ImageDataGenerator é um gerador de dados de imagem

# definição do modelo
classifier = Sequential() 

# configurando os parametros da camada de convolução
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# otmizando o modelo
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# flatten
classifier.add(Flatten())

# full connection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# compilando o classificador
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# pre processamento dos dados de treino
train_datagen = ImageDataGenerator(rescale=1./255, 
                                   shear_range=0.2, 
                                   zoom_range=0.2, 
                                   horizontal_flip=True)

# pre processamento dos dados de teste
test_datagen = ImageDataGenerator(rescale=1./255)

# carregando os dados de treino
train_set = train_datagen.flow_from_directory('./data/training_set',
                                                target_size=(64, 64),
                                                batch_size=40,
                                                class_mode='binary')

# carregando os dados de teste
test_set = test_datagen.flow_from_directory('./data/test_set',
                                             target_size=(64, 64),
                                             batch_size=40,
                                             class_mode='binary')

# treinamento do classificador
classifier.fit_generator(train_set,
                         steps_per_epoch=200,     # quantidade de passos por epoca(tamanho do conjunto de treinamento / batch_size)
                         epochs=5,                # quantidade de epocas
                         validation_data=test_set,
                         validation_steps=50)
                         