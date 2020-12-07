"""
file iterator for image semantic segmentation
"""
import time
from PIL import Image
from operator import itemgetter
import os
import os.path as osp

import numpy as np
import numpy.random as npr
import copy

from mxnet.io import DataBatch, DataIter
from mxnet.ndarray import array

from util.io import BatchFetcherGroup
from util.sampler import FixedSampler, RandomSampler
from util.util import get_interp_method, load_image_with_cache

def parse_split_file(dataset, split, num_sel_source = 500, num_source = 500, seed_int = 0, dataset_tgt='', split_tar='', data_root='', data_root_tgt='',gpus='0'):
    split_filename = 'issegm/data_list/{}/{}.lst'.format(dataset, split)
    image_list = []
    label_list = []
    origin_list = [] # record the origin of the image. 0 for source domain and 1 for target domain.
    # count0 = 0
    with open(split_filename) as f:
        for item in f.readlines():
            fields = item.strip().split('\t')
            image_list.append(os.path.join(data_root, fields[0]))
            label_list.append(os.path.join(data_root, fields[1]))
            origin_list.append(0)
    np.random.seed(seed_int)
    sel_idx = list( np.random.choice(num_source, num_sel_source, replace=False) )
    image_list = list( itemgetter(*sel_idx)(image_list) )
    label_list = list( itemgetter(*sel_idx)(label_list) )
    origin_list = list(itemgetter(*sel_idx)(origin_list))

    if not dataset_tgt == '':
        split_filename_tgt = 'issegm/data_list/{}/{}_training_gpu{}.lst'.format(dataset_tgt, split_tar,gpus)
        with open(split_filename_tgt) as f:
            for item in f.readlines():
                fields = item.strip().split('\t')
                image_list.append(os.path.join(data_root_tgt, fields[0]))
                label_list.append(os.path.join(fields[1]))
                origin_list.append(1)
    return image_list, label_list, origin_list

def make_divisible(v, divider):
    return int(np.ceil(float(v) / divider) * divider)

def _make_dirs(path):
    if not osp.isdir(path):
        os.makedirs(path)

class FileIter(DataIter):
    """FileIter object for image semantic segmentation.
    Parameters
    ----------

    dataset : string
        dataset
    split : string
        data split
        the list file of images and labels, whose each line is in the format:
        image_path \t label_path
    data_root : string
        the root data directory
    data_name : string
        the data name used in the network input
    label_name : string
        the label name used in SoftmaxOutput
    sampler: str
        how to shuffle the samples per epoch
    has_gt: bool
        if there are ground truth labels
    batch_images : int
        the number of images per batch
    meta : dict
        dataset specifications
    
    prefetch_threads: int
        the number of prefetchers
    prefetcher_type: string
        the type of prefechers, e.g., process/thread
    """
    def __init__(self,
                 dataset,
                 split,
                 data_root,
                 num_sel_source = 50,
                 num_source=100,
                 seed_int=0,
                 dataset_tgt = '',
                 split_tgt = '',
                 data_root_tgt = '',
                 data_name = 'data',
                 label_name = 'softmax_label',
                 sampler = 'fixed',
                 has_gt = True,
                 batch_images = 1,
                 meta = None,
                 ####
                 # transforming by the old fasion reader
                 rgb_mean = None, # (R, G, B)
                 feat_stride = 32,
                 label_stride = 32,
                 label_steps = 1,
                 origin_size = None,
                 origin_size_tgt=None,
                 crop_size = 0,
                 scale_rate_range = None,
                 crops_per_image = 1,
                 # or the new functional reader
                 transformer = None,
                 # image only pre-processing such as rgb scale and meanstd
                 transformer_image = None,
                 ####
                 prefetch_threads = 1,
                 prefetcher_type = 'thread',):
        super(FileIter, self).__init__()
        assert crop_size > 0
        
        self._meta = meta
        self._seed_int = seed_int
        self._data_name = data_name
        self._label_name = label_name
        self._has_gt = has_gt
        self._batch_images = batch_images
        self._feat_stride = feat_stride
        self._label_stride = label_stride
        self._label_steps = label_steps
        self._origin_size = origin_size
        self._origin_size_tgt = origin_size_tgt
        self._crop_size = make_divisible(crop_size, self._feat_stride)
        self._crops_per_image = crops_per_image
        #
        self._data_mean = None if rgb_mean is None else rgb_mean.reshape((1, 1, 3))
        self._scale_rate_range = (1.0, 1.0) if scale_rate_range is None else scale_rate_range
        #
        self._transformer = transformer
        self._transformer_image = transformer_image
        self._reader = self._read if self._transformer is None else self._read_transformer
        self._gpus = meta['gpus']
        
        self._ignore_label = 255

        self._image_list, self._label_list, self._origin_list = parse_split_file(dataset, split, num_sel_source,
                                                                                 num_source, self._seed_int, dataset_tgt,
                                                                                 split_tgt, data_root, data_root_tgt,self._gpus)
        self._perm_len = len(self._image_list)
        if sampler == 'fixed':
            sampler = FixedSampler(self._perm_len)
        elif sampler == 'random':
            sampler = RandomSampler(self._perm_len)
        
        assert self._label_steps == 1
        assert self._crops_per_image == 1
        self.batch_size = self._batch_images
        
        self._cache = {} if self._meta['cache_images'] else None
        
        self._fetcher = BatchFetcherGroup(self,
                                          sampler,
                                          batch_images,
                                          prefetch_threads,
                                          prefetch_threads*2,
                                          prefetcher_type)
        
        if crop_size > 0:
            crop_h = crop_w = self._crop_size
        else:
            rim = load_image_with_cache(self._image_list[0], self._cache)
            crop_h = make_divisible(rim.size[1], self._feat_stride)
            crop_w = make_divisible(rim.size[0], self._feat_stride)
        self._data = list({self._data_name: np.zeros((1, 3, crop_h, crop_w), np.single)}.items())
        self._label = list({self._label_name: np.zeros((1, crop_h * crop_w / self._label_stride**2), np.single)}.items())

    def read(self, db_inds):
        return self._reader(db_inds)
    
    def _read_transformer(self, db_inds):
        output_list = []
        output_shape = [0, 0]
        for db_ind in db_inds:
            # load an image
            rim = load_image_with_cache(self._image_list[db_ind], self._cache).convert('RGB')
            data = np.array(rim, np.uint8)
            # load the label
            if self._has_gt:
                rlabel = load_image_with_cache(self._label_list[db_ind], self._cache)
                label = np.array(rlabel, np.uint8)
            else:
                label = self._ignore_label * np.ones(data.shape[:2], np.uint8)
            # jitter
            if self._transformer is not None:
                data, label = self._transformer(data, label)
            lsy = lsx = self._label_stride / 2
            label = label[lsy::self._label_stride, lsx::self._label_stride]
            output_list.append((data, label))
            output_shape = np.maximum(output_shape, data.shape[:2])
        
        output_shape = [make_divisible(_, self._feat_stride) for _ in output_shape]
        output = [np.zeros((self.batch_size, 3, output_shape[0], output_shape[1]), np.single),
                  self._ignore_label * np.ones((self.batch_size, output_shape[0]/self._label_stride, output_shape[1]/self._label_stride), np.single),]
        for i in xrange(len(output_list)):
            imh, imw = output_list[i][0].shape[:2]
            output[0][i][:, :imh, :imw] = output_list[i][0].transpose(2, 0, 1)
            output[1][i][:imh, :imw] = output_list[i][1]
        output[1] = output[1].reshape((self.batch_size, -1))
            
        return tuple(output)
            
    def _read(self, db_inds):
        label_2_id_src = self._meta['label_2_id_src']
        label_2_id_tgt = self._meta['label_2_id_tgt']
        mine_port = self._meta['mine_port']
        mine_id = self._meta['mine_id']
        mine_id_priority = self._meta['mine_id_priority']
        ##

        target_crop_size = self._crop_size
        label_size = target_crop_size // self._label_stride
        assert label_size * self._label_stride == target_crop_size
        label_per_image = label_size**2
        locs_per_crop = self._label_steps ** 2
        output = []
        for _ in xrange(locs_per_crop * self._crops_per_image):
            output_data = np.zeros((self._batch_images, 3, target_crop_size, target_crop_size), np.single)
            output_label = np.zeros((self._batch_images, label_per_image), np.single)
            output.append([output_data, output_label])
        for i,db_ind in enumerate(db_inds):
            if self._origin_list[db_ind] == 0:
                max_h, max_w = self._meta['max_shape_src']
                target_size_range = [int(_ * self._origin_size) for _ in self._scale_rate_range]
                min_rate = 1. * target_size_range[0] / max(max_h, max_w)
                max_crop_size = int(target_crop_size / min_rate)
            elif self._origin_list[db_ind] == 1:
                max_h, max_w = self._meta['max_shape_tgt']
                target_size_range = [int(_ * self._origin_size_tgt) for _ in self._scale_rate_range]
                min_rate = 1. * target_size_range[0] / max(max_h, max_w)
                max_crop_size = int(target_crop_size / min_rate)
            # load an image
            im = np.array(load_image_with_cache(self._image_list[db_ind], self._cache).convert('RGB'))
            h, w = im.shape[:2]
            target_size = npr.randint(target_size_range[0], target_size_range[1] + 1)
            #######################
            rate = 1. * target_size / max(max_h, max_w)  
            #######################
            crop_size = int(target_crop_size / rate)
            label_stride = self._label_stride / rate
            # make sure there is a all-zero border
            d0 = max(1, int(label_stride // 2))
            # allow shifting within the grid between the used adjacent labels
            d1 = max(0, int(label_stride - d0))
            # prepare the image
            ##########
            nim_size = max(max_crop_size, h, w) + d1 + d0
            ##########
            nim = np.zeros((nim_size, nim_size, 3), np.single)
            nim += self._data_mean
            nim[d0:d0 + h, d0:d0 + w, :] = im
            # label
            nlabel = self._ignore_label * np.ones((nim_size, nim_size), np.uint8)
            ##########
            label = np.array(load_image_with_cache(self._label_list[db_ind], self._cache))
            # label = olabel[shift_h:shift_h + h, shift_w:shift_w + w]
            ##########
            if self._origin_list[db_ind] == 0:
                label = label_2_id_src[label]
            elif self._origin_list[db_ind] == 1:
                label = label_2_id_tgt[label]
            nlabel[d0:d0 + h, d0:d0 + w] = label
            # crop
            real_label_stride = label_stride / self._label_steps
            mine_flag = npr.rand(1) < mine_port
            sel_mine_id = 0
            ##########  few class patch mining
            if mine_flag:
                mlabel = self._ignore_label * np.ones((nim_size, nim_size), np.uint8)
                mlabel[d0+int(crop_size/2):max(1, real_label_stride, d0 + h - int(crop_size/2) - 1 ),d0+int(crop_size/2):max(1,real_label_stride,d0+w-int(crop_size/2) - 1)] = nlabel[d0+int(crop_size/2):max(1,real_label_stride,d0+h-int(crop_size/2) - 1),d0+int(crop_size/2):max(1,real_label_stride,d0+w-int(crop_size/2) - 1)]
                label_unique = np.unique(mlabel)
                mine_id_priority_temp = np.array([a for a in mine_id_priority if a in label_unique])
                if mine_id_priority_temp.size != 0:
                    mine_id_exist = mine_id_priority_temp
                else:
                    mine_id_temp = np.array([a for a in mine_id if a in label_unique])
                    mine_id_exist = np.array([b for b in mine_id_temp])
                mine_id_exist_size = mine_id_exist.size
                if mine_id_exist_size == 0:
                    sy = npr.randint(0, max(1, real_label_stride, d0 + h - crop_size + 1), self._crops_per_image)
                    sx = npr.randint(0, max(1, real_label_stride, d0 + w - crop_size + 1), self._crops_per_image)
                else:
                    sel_mine_loc = int( np.floor(npr.uniform(0,mine_id_exist_size)) )
                    sel_mine_id = mine_id_exist[sel_mine_loc]
                    mine_id_loc = np.where(mlabel == sel_mine_id) # tuple
                    mine_id_len = len(mine_id_loc[0])
                    seed_loc = npr.randint(0, mine_id_len,self._crops_per_image)
                    if self._crops_per_image == 1:
                        sy = mine_id_loc[0][seed_loc]
                        sx = mine_id_loc[1][seed_loc]
                    else:
                        sy = int(np.ones(self._crops_per_image))
                        sx = int(np.ones(self._crops_per_image))
                        for i in np.arange(self._crops_per_image):
                            sy[i] = mine_id_loc[0][seed_loc[i]]
                            sx[i] = mine_id_loc[1][seed_loc[i]]
            ########## few class patch mining
            else:
                sy = npr.randint(0, max(1, real_label_stride, d0 + h - crop_size + 1), self._crops_per_image)
                sx = npr.randint(0, max(1, real_label_stride, d0 + w - crop_size + 1), self._crops_per_image)
            dyx = np.arange(0, label_stride, real_label_stride).astype(np.int32)[:self._label_steps].tolist()
            dy = dyx * self._label_steps
            dx = sum([[_] * self._label_steps for _ in dyx], [])
            for k in xrange(self._crops_per_image):
                do_flipping = npr.randint(2) == 0
                for j in xrange(locs_per_crop):
                    # cropping & resizing image
                    if mine_flag and mine_id_exist_size != 0:
                        remain_minus = crop_size - int(crop_size / 2)
                        tim = nim[sy[k] + dy[j] - int(crop_size / 2):sy[k] + dy[j] + remain_minus ,sx[k] + dx[j] - int(crop_size / 2):sx[k] + dx[j] + remain_minus, :].astype(np.uint8)
                        assert tim.shape[0] == tim.shape[1] == crop_size
                        tlabel = nlabel[sy[k] + dy[j] - int(crop_size / 2):sy[k] + dy[j] + remain_minus,sx[k] + dx[j] - int(crop_size / 2):sx[k] + dx[j] + remain_minus]
                        assert tlabel.shape[0] == tlabel.shape[1] == crop_size
                    else:
                        tim = nim[sy[k] + dy[j]:sy[k] + dy[j] + crop_size, sx[k] + dx[j]:sx[k] + dx[j] + crop_size,:].astype(np.uint8)
                        assert tim.shape[0] == tim.shape[1] == crop_size
                        tlabel = nlabel[sy[k] + dy[j]:sy[k] + dy[j] + crop_size,sx[k] + dx[j]:sx[k] + dx[j] + crop_size]
                        assert tlabel.shape[0] == tlabel.shape[1] == crop_size

                    interp_method = get_interp_method(crop_size, crop_size, target_crop_size, target_crop_size)
                    rim = Image.fromarray(tim).resize((target_crop_size,target_crop_size), interp_method)
                    rim = np.array(rim)
                    # cropping & resizing label
                    rlabel = Image.fromarray(tlabel).resize((target_crop_size,target_crop_size), Image.NEAREST)
                    slabel = np.array(rlabel).copy()
                    lsy = self._label_stride / 2
                    lsx = self._label_stride / 2
                    rlabel = np.array(rlabel)[lsy : target_crop_size : self._label_stride, lsx : target_crop_size : self._label_stride]
                    # flipping
                    if do_flipping:
                        rim = rim[:, ::-1, :]
                        rlabel = rlabel[:, ::-1]
		    # for debug
                    #output_patch = 'patch_debug/'
                    #_make_dirs(output_patch)
                    #save_time = str(time.time())
                    #output_patch_cls = osp.join(output_patch,str(sel_mine_id))
                    #_make_dirs(output_patch_cls)
                    #sample_name = osp.splitext(osp.basename(self._image_list[db_ind]))[0]
                    #Image.fromarray(rim).save(osp.join(output_patch_cls,sample_name + '_img.png'))
                    #Image.fromarray(slabel).save(osp.join(output_patch_cls,sample_name + '_label.png'))
                    # transformers
                    if self._transformer_image is not None:
                        rim = self._transformer_image(rim)
                    else:
                        rim -= self._data_mean
                    # assign
                    output[k*locs_per_crop+j][0][i,:] = rim.transpose(2,0,1)
                    output[k*locs_per_crop+j][1][i,:] = rlabel.flatten()
        return output
    
    @property
    def batch_images(self):
        return self._batch_images
    
    @property
    def batches_per_epoch(self):
        return self._perm_len // self._batch_images

    @property
    def provide_data(self):
        """The name and shape of data provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self._data]

    @property
    def provide_label(self):
        """The name and shape of label provided by this iterator"""
        return [(k, tuple([self.batch_size] + list(v.shape[1:]))) for k, v in self._label]

    def reset(self):
        self._fetcher.reset()

    def next(self):
        if self._fetcher.iter_next():
            # tic = time.time()
            data_batch = self._fetcher.get()
            # print 'Waited for {} seconds'.format(time.time() - tic)
        else:
            raise StopIteration
        
        return DataBatch(data=[array(data_batch[0])], label=[array(data_batch[1])])

    def debug(self):
        for i in xrange(self._perm_len):
            self.read([i])
            print 'Done {}/{}'.format(i+1, self._perm_len)

