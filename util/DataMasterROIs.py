import dicom
import time
import numpy as np
from PIL import Image
import csv
#import BatcherROIs as Batcher
from util import BatcherROIs as Batcher
import collections
import json

root_dir =  "/deep/group/med/images/DDSM_ROI"

class DataMaster:
    '''
    csv data loader and processor, tells Batcher
    where to look for data
    '''
    def load_csv(self, filename):
        '''
        your most basic csv loader
        '''
        with open(filename, 'r') as file:
            file = csv.reader(file, delimiter=',')
            file = list(file)
        return file

    def get_metadata(self):
        ''' 
        loads image metadata
        we need this to index into the file structure because
        the metadata for each image also contains path information
        right now we are mixing the mass and calc data
        '''
        mass_files = self.load_csv(self.mass_filename)
        calc_files = self.load_csv(self.calc_filename)
        # extract the headers and turn them into dicts
        mass_headers = mass_files[0]
        calc_headers = calc_files[0]
        del mass_files[0]
        del calc_files[0]

        self.num_calc = len(calc_files)
        self.num_mass = len(mass_files)

        self.mass_headers = {item[1]: item[0] for item in enumerate(mass_headers)}
        self.calc_headers = {item[1]: item[0] for item in enumerate(calc_headers)} 

        # now we can preprocess data easier
        self.patient_ids = list(set([p[0] for p in calc_files] + [p[0] for p in mass_files]))
        return mass_files + calc_files

    def extract_int(self, s):
        try:
            for elem in s:
                int(elem)
            return True
        except ValueError:
            return False

    def preprocess(self):
        '''
        we need to discretize the text label attributes
        processing in this respect will be done here
        '''
        # for each field, there's gonna be discrete categories that we need to turn into numbers
        '''
        to_pack{
            pathology: {0, 1, 2}
            breast
        }
        '''
        mass_attrib = collections.defaultdict(set)
        calc_attrib = collections.defaultdict(set)
        mass_fields = ['breast density', 'left or right breast', 'image view', 'abnormality id', 'mass shape', 'mass margins', 'assessment', 'pathology', 'subtlety']
        calc_fields = ['breast density', 'left or right breast', 'image view', 'abnormality id', 'assessment', 'pathology', 'subtlety', 'calc type', 'calc distribution']
        for i, row in enumerate(self.metadata):
            if i <= self.num_mass: # CALC
                for field in mass_fields:
                    mass_attrib[field].add(row[self.mass_headers[field]])
            else:
                for field in calc_fields:
                    calc_attrib[field].add(row[self.calc_headers[field]])

        for field in mass_attrib:
            enum_list = list(mass_attrib[field])
            if self.extract_int(mass_attrib[field]):
                mass_attrib[field] = {item[1]: int(item[1]) for item in enumerate(enum_list)}
            else:
                mass_attrib[field] = {item[1]: item[0] + 1 for item in enumerate(enum_list)}

        for field in calc_attrib:
            enum_list = list(calc_attrib[field])
            if self.extract_int(calc_attrib[field]):
                calc_attrib[field] = {item[1]: int(item[1]) for item in enumerate(enum_list)}
            else:
                calc_attrib[field] = {item[1]: item[0] + 1 for item in enumerate(enum_list)}

        self.attr2onehot = {
            "pathology": mass_attrib['pathology'],
            'mass': mass_attrib,
            'calc': calc_attrib
        } 
        


    def get_train_val_inds(self):
        if self.locked_inds:
            file = open('util/train_test_split_' + str(self.k_folds), 'r')
            file_read = file.read()
            json_obj = json.loads(file_read)
            train_ids, val_ids = json_obj.values()
        else:
            patient_indices = np.asarray(range(len(self.patient_ids)))
            np.random.shuffle( np.asarray(patient_indices))
            
            ids = np.array_split(patient_indices, self.k_folds)
            val_patients = [self.patient_ids[i] for i in ids[-1]]
            train_patients = [self.patient_ids[i] for i in np.concatenate(ids[:-1])]

            full_data = np.asarray(range(len(self.metadata)))
            np.random.shuffle(full_data)
            train_ids = [index for index in full_data if self.metadata[index][0] in train_patients]
            val_ids = [index for index in full_data if self.metadata[index][0] in val_patients]
            print('train, test split: train {} - test {}'.format(len(train_ids), len(val_ids)))
        return train_ids, val_ids

    def next_fold(self):
        '''
        sets up and returns a batcher on the next validation fold
        returns the (train, test) batchers on the next fold
        '''         
        train_inds, test_inds = self.get_train_val_inds()
        # return two Batcher
        train_mean = 0
        train_std = 0
        if self.new_batch:
            print('calculating train mean')
            train_fp = Batcher.Batcher(
                self.batch_sz, 
                self.metadata, 
                train_inds, 
                self.mass_headers, 
                self.calc_headers, 
                root_dir, 
                self.attr2onehot, 
                new_batch=self.new_batch
            )
            test_fp = Batcher.Batcher(
                self.batch_sz, 
                self.metadata, 
                test_inds, 
                self.mass_headers, 
                self.calc_headers, 
                root_dir, 
                self.attr2onehot, 
                new_batch=self.new_batch
            )
            train_mean, train_std = train_fp.get_train_stats() 
            _, _ = test_fp.get_train_stats()
        train = Batcher.Batcher(
            self.batch_sz, 
            self.metadata, 
            train_inds, 
            self.mass_headers, 
            self.calc_headers, 
            root_dir, 
            self.attr2onehot,  
            mean=train_mean, 
            std=train_std,
            new_batch=self.new_batch
        )
        test = Batcher.Batcher(
            self.batch_sz, 
            self.metadata, 
            test_inds, 
            self.mass_headers, 
            self.calc_headers, 
            root_dir, 
            self.attr2onehot, 
            mean=train_mean,
            std=train_std,
            new_batch=self.new_batch
        )
        print(train_mean, train_std)
        self.curr_fold += 1
        return train, test

    def __init__(self, batch_sz, k_folds, new_batch=False, locked_inds=True):
        # instance variables
        self.mass_filename = 'mass_case_description_train_set.csv'#root_dir + '/mass_case_description_train_set.csv'
        self.calc_filename = 'calc_case_description_train_set.csv'#root_dir + '/calc_case_description_train_set.csv'
        self.curr_fold = 0
        self.k_folds = k_folds
        self.batch_sz = batch_sz
        self.mass_headers = {}
        self.calc_headers = {}
        self.new_batch = new_batch
        self.locked_inds = locked_inds
        # first, load all the image metadata contained in the csv files
        self.metadata = self.get_metadata() 
        # second, preprocess all the attributes
        self.preprocess()
        # third, shuffle our dataset into folds
        print("There are total of {} data points".format(len(self.metadata)))
        
def massage_data():
    dm = DataMaster(20, 10, new_batch=True, locked_inds=True)
    tr, te = dm.next_fold()
    gen_train = tr.get_iterator()
    gen_test = te.get_iterator()
    for tup in gen_train:
        print(tup[0].shape)
        #print(tup[1])
        #print(tup[2])
        print("----") 
    for tup in gen_test:
        print(tup[0].shape)
        print("----")

if __name__ == '__main__':
    dm = DataMaster(20, 10, new_batch=True, locked_inds=True)
    tr, te = dm.next_fold()
    gen = te.get_iterator()
    for tup in gen:
        print(tup[0].shape)
        print(tup[1])
        #print(tup[2])
        print("----")
