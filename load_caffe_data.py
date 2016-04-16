from timer import timer
from data_sets import create_data_set


import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import re
import math
from utility import array_functions
from utility import helper_functions
from data import data as data_class

def run_main():
    import caffe
    adience_caffe_model_dir = 'C:\\Users\\Aubrey\\Desktop\\cnn_age_gender_models_and_data.0.0.2\\'

    age_net_pretrained='/age_net.caffemodel'
    age_net_model_file='/deploy_age.prototxt'

    age_net = caffe.Classifier(adience_caffe_model_dir + age_net_model_file,
                               adience_caffe_model_dir + age_net_pretrained,
                               channel_swap=(2,1,0),
                               raw_scale=255,
                               image_dims=(256, 256))

    age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']

    adience_image_dir = 'C:\\Users\\Aubrey\\Desktop\\adience_aligned\\aligned\\'
    adience_metadata_file = 'C:\\Users\\Aubrey\\Desktop\\adience_aligned\\alined_metadata\\all_photos.csv'

    metadata = create_data_set.load_csv(adience_metadata_file,
                                         dtype='string',
                                         delim='\t',
                                         )

    column_names = metadata[0].tolist()
    photo_data = metadata[1]
    face_id_col = column_names.index('face_id')
    user_id_col = column_names.index('user_id')
    image_name_col = column_names.index('original_image')
    age_col = column_names.index('age')
    x = np.zeros((photo_data.shape[0], 512))
    y = np.zeros((photo_data.shape[0]))
    id = np.zeros((photo_data.shape[0]))
    i = 0
    last_perc_done = 0
    for idx, row in enumerate(photo_data):
        perc_done = math.floor(100 * float(idx) / len(photo_data))
        if perc_done > last_perc_done:
            last_perc_done = perc_done
            print str(perc_done) + '% done'
        image_dir = adience_image_dir + row[user_id_col] + '/'
        face_id = row[face_id_col]
        '''
        images_in_dir = os.listdir(image_dir)
        matching_images = [s for s in images_in_dir if s.find(row[image_name_col]) >= 0]
        assert len(matching_images) < 2
        if len(matching_images) == 0:
            print 'Skipping: ' + image
            continue
        '''
        image = image_dir + 'landmark_aligned_face.' + str(face_id) + '.' + row[image_name_col]
        if not os.path.isfile(image):
            print 'Skipping: ' + image
            continue
        input_image = caffe.io.load_image(image)
        age = row[age_col]
        blobs = ['fc7']
        features_age = predict_blobs(age_net,[input_image],blobs)
        x[i,:] = features_age
        y[i] = extract_age(age)
        id[i] = float(face_id)
        i += 1
    data = data_class.Data()
    data.x = x
    data.instance_ids = id
    data.y = y
    data.is_regression = True
    data.set_train()
    data.set_target()
    data.set_true_y()
    data_file = create_data_set.adience_aligned_cnn_file
    helper_functions.save_object('data_sets/' + data_file, data)
    print 'TODO'

def extract_age(age_str):
    age = 0
    age_str = re.sub('[(),]', ' ', age_str)
    try:
        age = float(age_str)
    except:
        age_range = [float(s) for s in age_str.split() if s.isdigit()]
        age = np.asarray(age_range).mean()
    return age

def subset_1_per_instance_id():
    data = helper_functions.load_object('data_sets/' + create_data_set.adience_aligned_cnn_file)
    to_keep = array_functions.false(data.n)
    all_ids = np.unique(data.instance_ids)
    for id in all_ids:
        has_id = (data.instance_ids == id).nonzero()[0]
        to_keep[has_id[0]] = True
        pass
    to_keep = to_keep & data.is_labeled
    data = data.get_subset(to_keep)
    helper_functions.save_object('data_sets/' + create_data_set.adience_aligned_cnn_1_per_instance_id_file,
                                 data)
    pass

def predict_blobs(self, inputs, blobs=[]):
    """
    extension to classifier prediction function that also returns blobs
    """
    # Scale to standardize input dimensions.
    assert len(blobs) == 1
    input_ = np.zeros((len(inputs),
                       self.image_dims[0],
                       self.image_dims[1],
                       inputs[0].shape[2]),
                      dtype=np.float32)
    for ix, in_ in enumerate(inputs):
        input_[ix] = caffe.io.resize_image(in_, self.image_dims)

    # Take center crop.
    center = np.array(self.image_dims) / 2.0
    crop = np.tile(center, (1, 2))[0] + np.concatenate([
        -self.crop_dims / 2.0,
        self.crop_dims / 2.0
    ])
    input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

    # Classify
    caffe_in = np.zeros(np.array(input_.shape)[[0, 3, 1, 2]],
                        dtype=np.float32)
    for ix, in_ in enumerate(input_):
        caffe_in[ix] = self.transformer.preprocess(self.inputs[0], in_)
    out = self.forward_all(blobs=blobs,**{self.inputs[0]: caffe_in})
    predictions = out[blobs[0]]

    return predictions

if __name__ == '__main__':
    #run_main()
    subset_1_per_instance_id()