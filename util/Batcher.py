import dicom
import numpy as np
from PIL import Image
import csv
import os
from scipy.misc import imresize, imsave
import matplotlib.pyplot as plt

class Batcher:
    def __init__(self, batch_sz, metadata, indices, mass_headers, calc_headers, root, attr2onehot, mean=0, new_batch=False):
        '''
        This batcher takes in rows of metadata formatted
        specifically for DDSM images with the image directory
        structure downloaded from TCIA
        '''
        self.batch_sz = batch_sz
        self.metadata = metadata
        self.mass_headers = mass_headers
        self.calc_headers = calc_headers
        self.indices = indices
        self.root = root
        self.attr2onehot = attr2onehot
        self.mean = mean
        self.new_batch = new_batch
    
    def visualize(self, img):
        '''
        debug tool: visualize the grey scale image for sanity check.
        '''
        plt.imshow(img, cmap='gray')
        plt.show()

    def to_uint8(self, img):
        '''
        converts an image from uint16 to uint8 format.
        uses lookup table to reduce memory usage
        also more efficient
        '''
        lut = np.arange(2**16, dtype='uint16')
        lut -= np.amin(img)
        lut //= int((np.amax(img) - np.amin(img) + 1) / 256)
        return np.take(lut, img)

    def resize(self, img, H = 250, W = 250):
        '''
        resize all images to 4000 x 3000 to be batchable
        '''
        return imresize(img, (H, W))

    def get_train_mean(self):
        '''
        first pass of images to get mean pixel value for preprocessing
        '''
        it = self.get_iterator(mean_process=True) 
        mean = 0.0
        counter = 0
        for imgs, labels, _, paths in it:
            for im, image_path in zip(imgs, paths):
                counter += 1
                im = self.to_uint8(im)
                im = self.resize(im)
                print('saving {} inside get_train_mean'.format(image_path))
                np.save(image_path, im)
                H, W = im.shape[0], im.shape[1]
                # incremental mean update for numerical stability
                mean += (np.sum(im) - mean * H * W) / (counter * H * W) 
        return mean

    def preprocess(self, img, unseen=False):
        '''
        Preprocessing step:
        convert to 8 bit, resize to H X W, reduce by mean
        '''
        if unseen:
            img = self.to_uint8(img)
            img = self.resize(img)
        img = img.astype(np.float64)
        if self.mean != 0:  img -= self.mean 
        return img

    def get_image_from_path(self, path):
        path += '/' + next(os.walk(os.path.expanduser(path)))[1][0]
        path += '/' + next(os.walk(os.path.expanduser(path)))[1][0]
        path += '/' + next(os.walk(os.path.expanduser(path)))[2][0]
        # 4. read image from DICOM format into 16bit pixel value
        DCM_img = dicom.read_file(path)
        img = np.asarray(DCM_img.pixel_array)
        return img

    def generate_attribute(self, row, is_mass):
        generic_field =  ['breast_density', 'abn_num','assessment', 'subtlety']
        mass_fields = ['mass_shape', 'mass_margins']
        calc_fields = ['calc_type', 'calc_distribution']
        attribute = [] #[0] * 4

        # mass: shape 10, margin 7
        # calc: type 15, distrib 6
        # feat1: 17 feat2: 21
        if is_mass:
            for field in generic_field:
                attribute.append(self.attr2onehot['mass'][field][row[self.mass_headers[field]]])
            for field, pad in zip(mass_fields, [15, 6]):
                mass_feature = [0] * len(self.attr2onehot['mass'][field])
                parts = row[self.mass_headers[field]].split('-')
                for part in parts:
                    mass_feature[self.attr2onehot['mass'][field][part] - 1] = 1
                #mass_feature[self.attr2onehot['mass'][field][row[self.mass_headers[field]]] - 1] = 1
                attribute += mass_feature
                attribute += [0] * pad
        else:
            for field in generic_field:
                attribute.append(self.attr2onehot['calc'][field][row[self.calc_headers[field]]])
            for field, pad in zip(calc_fields, [10, 7]):
                attribute += [0] * pad
                calc_feature = [0] * len(self.attr2onehot['calc'][field])
                parts = row[self.calc_headers[field]].split('-')
                for part in parts:
                    calc_feature[self.attr2onehot['calc'][field][part] - 1] = 1
                #calc_feature[self.attr2onehot['calc'][field][row[self.calc_headers[field]]] - 1] = 1
                attribute += calc_feature

        return attribute
    
    def get_iterator(self, mean_process=False):
        '''
        Data iterator. get_iterator returns all batches
        in the form of (X, y) tuples
        '''
        X = []
        y = []
        attributes = []
        new_image_flag = False
        paths = []
        counter = 0
        for i in range(len(self.indices)):
            
            row = self.metadata[self.indices[i]]
            path = self.root + '/'
            # 1. figure out if this image is a mass or a calc
            if 'Mass' in row[self.mass_headers['od_img_path']]:
                path += 'Mass-Training'
            else:
                path += 'Calc-Training'
            # 2. build the image path
            path += '_' + row[self.mass_headers['patient_id']] \
                + '_' + row[self.mass_headers['side']] + '_' \
                + row[self.mass_headers['view']]

            if not os.path.exists(path):
                continue
            # 3. wade through two layers of useless directories
            down_a_level = next(os.walk(os.path.expanduser(path)))
            image_name = 'image_{}'.format(row[self.mass_headers['patient_id']])
            image_path = path + '/' + image_name
            # if we're trying to just get images for mean calculations
            if mean_process:
                img = self.get_image_from_path(path)
            elif self.new_batch:
                # this means we're relying on mean-processed images to do further processing on
                try:
                    img = np.load(image_path + '.npy')
                except:
                    raise Exception('Most likely a file read error, or that somehow we tried to read a file we haven\'t preprocessed before')
                # no unseen flag because we've already seen these images!
                img = self.preprocess(img)
                print('saving {} inside get_iterator'.format(image_path))
                np.save(image_path, img)
            else:
                # we are assuming that all the images have been processed before
                try:
                    # print('opening completely preprocessed {} inside get_iterator'.format(image_path))
                    img = np.load(image_path + '.npy')
                except:
                    img = self.get_image_from_path(path)
                    img = self.preprocess(img, unseen=True)
                    print('saving {} inside get_iterator'.format(image_path))
                    np.save(image_path, img)
            # 6. add the image to the batch 
            X.append(img)
            #imsave("out/{}.png".format(row[0]), img)
            # 7. do some mojo with the label and append to y
            # probably gonna be one hot or something
            label = self.attr2onehot['pathology'][row[self.mass_headers['pathology']]] - 1
            y.append(label)

            # 8. Get those attributes
            if 'Mass' in row[self.mass_headers['od_img_path']]: 
                attributes.append(self.generate_attribute(row, True))
            else:
                attributes.append(self.generate_attribute(row, False))
            # only do this if we are trying to make a new batch, and if we are catering to get_train_mean()
            if mean_process:
                paths.append(image_path)
            # 8. check if our batch is ready
            counter += 1
            if counter >= self.batch_sz: 
                if mean_process:
                    yield (np.asarray(X), np.asarray(y), np.asarray(attributes), paths)
                else:
                    yield (np.asarray(X), np.asarray(y), np.asarray(attributes))
                X = []
                y = []
                paths = []
                attributes = []
                counter = 0
        if not X or not y:
            return

        if mean_process:
            yield (np.asarray(X), np.asarray(y), np.asarray(attributes), paths)
        else:
            X_out = np.asarray(X)
            y_out = np.asarray(y)
            attrib_out = np.asarray(attributes)
            yield (X_out, y_out, attrib_out)
