import sys
import argparse
import numpy as np
from keras.utils import np_utils
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator

from utils import load_data, shuffle
from model import CNN, VGG10, inception, DNN, n_classes

parser = argparse.ArgumentParser(prog='main.py', description='ML2017-hw3 Emotion Image Classifier')
parser.add_argument('--test', action='store_true', default=False)
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--batch', type=int, default=512)
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--model', type=str, default='VGG10')
parser.add_argument('--tt_file', type=str, default='../data/test.csv')
parser.add_argument('--tr_file', type=str, default='../data/train.csv')
parser.add_argument('--prefixion', type=str, default='unnamed')
parser.add_argument('--params_dir', type=str, default='./')
parser.add_argument('--output_file', type=str, default='./hw3.csv')
parser.add_argument('--valid_ratio', type=float, default=0.1)
args = parser.parse_args()


paramfile = args.params_dir + args.prefixion + '.h5'
loggerfile =  args.params_dir + args.prefixion + '_logger.csv'

if args.test == False and args.train == False:
    print ('Error: can\'t find argument --train or --test.')
    sys.exit()

print('\nSelect %s model...' % args.model)
if args.model == 'VGG10':
    model = VGG10()
if args.model == 'CNN':
    model = CNN()
if args.model == 'inception':
    model = inception()
if args.model == 'DNN':
    model = DNN()

print('\nLoad data...')
if args.test == True:
    test_x, _ = load_data(args.tt_file)
    if args.model != 'DNN':
        test_x = np.reshape(test_x, (test_x.shape[0], 48, 48, 1))


if args.train == True:
    train_x, train_y = load_data(args.tr_file)
    if args.model != 'DNN':
        train_x = np.reshape(train_x, (train_x.shape[0], 48, 48, 1))
    train_y = np_utils.to_categorical(train_y, n_classes)
    
    if args.valid_ratio != 0.0:
        valid_size = int((train_x.shape[0] * args.valid_ratio) // 1)
        print('Split %d validation\n' % valid_size)
        train_x, train_y = shuffle(train_x, train_y)
        valid_x, valid_y = train_x[:valid_size], train_y[:valid_size]
        train_x, train_y = train_x[valid_size:], train_y[valid_size:]
        
    

    datagen = ImageDataGenerator(
                shear_range = 0.2,
                rotation_range = 15,
                horizontal_flip = True,
                zoom_range = [0.8, 1.2],
                width_shift_range = 0.2,
                height_shift_range = 0.2,
            )
    
    num_batches = (train_x.shape[0] // args.batch) + 1
    csvlogger = CSVLogger(loggerfile)
    earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=1, mode='max')
    checkpoint = ModelCheckpoint(paramfile, monitor='val_acc', save_best_only=True, 
                                 save_weights_only=True, verbose=0, mode='max')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Start training...')
    if args.model != 'DNN':
        model.fit_generator(datagen.flow(train_x, train_y, batch_size=args.batch), steps_per_epoch=5*num_batches,
                            epochs=args.epoch, verbose=1, validation_data=(valid_x, valid_y), workers=8,
                            callbacks = [checkpoint, earlystopping, csvlogger])
    else:
        model.fit(train_x, train_y, batch_size=args.batch, epochs=args.epoch, verbose=1, validation_data=(valid_x, valid_y), 
                  shuffle=True, callbacks = [checkpoint, earlystopping, csvlogger])

    
if args.test == True:
    print('\nLoad model parameters...\n')
    model.load_weights(paramfile)

    print('Start testing...\n')
    pred = model.predict(test_x)
    pred = pred.argmax(axis=-1)

    print('Save prediction...\n')
    outfile = open(args.output_file, 'w')
    print('id,label', file=outfile)
    for i in range(len(pred)):
        print(i, pred[i], sep=',', file=outfile)
