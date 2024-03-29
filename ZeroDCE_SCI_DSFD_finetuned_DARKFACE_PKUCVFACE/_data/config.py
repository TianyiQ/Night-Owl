# config.py
import os.path

# gets home dir cross platform
HOME = os.path.expanduser('~')

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

widerface_640 = {
    'num_classes': 2,

     #'lr_steps': (80000, 100000, 120000),
     #'max_iter': 120000,
     'lr_steps': (40000, 50000, 60000),
     'max_iter': 60000,

    'feature_maps': [160, 80, 40, 20, 10, 5],
    'min_dim': 640,

    'steps': [4, 8, 16, 32, 64, 128],   # stride 
    
    'variance': [0.1, 0.2],
    'clip': True,  # make default box in [0,1]
    'name': 'WIDERFace',
    'l2norm_scale': [10, 8, 5],
    'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512] , 
    'extras': [256, 'S', 512, 128, 'S', 256],
    
    'mbox': [1, 1, 1, 1, 1, 1] , 
    #'mbox': [2, 2, 2, 2, 2, 2],
    #'mbox': [4, 4, 4, 4, 4, 4],
    'min_sizes': [16, 32, 64, 128, 256, 512],
    'max_sizes': [],
    #'max_sizes': [8, 16, 32, 64, 128, 256],
    #'aspect_ratios': [ [],[],[],[],[],[] ],   # [1,2]  default 1
    'aspect_ratios': [ [1.5],[1.5],[1.5],[1.5],[1.5],[1.5] ],   # [1,2]  default 1
    
    'backbone': 'resnet152' , # vgg, resnet, detnet, resnet50
    'feature_pyramid_network':True ,
    'bottom_up_path': False ,
    'feature_enhance_module': True ,
    'max_in_out': True , 
    'focal_loss': False ,
    'progressive_anchor': True ,
    'refinedet': False ,   
    'max_out': False , 
    'anchor_compensation': False , 
    'data_anchor_sampling': False ,
   
    'overlap_thresh' : [0.4] ,
    'negpos_ratio':3 , 
    # test
    'nms_thresh':0.3 ,
    'conf_thresh':0.01 ,
    'num_thresh':5000 ,
}

darkface_640 = {
    'num_classes': 2,

     #'lr_steps': (80000, 100000, 120000),
     #'max_iter': 120000,
     'lr_steps': (40000, 50000, 60000),
     'max_iter': 60000,

    'feature_maps': [160, 80, 40, 20, 10, 5],
    'min_dim': 640,

    'steps': [4, 8, 16, 32, 64, 128],   # stride 
    
    'variance': [0.1, 0.2],
    'clip': True,  # make default box in [0,1]
    'name': 'DARKFace',
    'l2norm_scale': [10, 8, 5],
    'base': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M', 512, 512, 512] , 
    'extras': [256, 'S', 512, 128, 'S', 256],
    
    'mbox': [1, 1, 1, 1, 1, 1] , 
    #'mbox': [2, 2, 2, 2, 2, 2],
    #'mbox': [4, 4, 4, 4, 4, 4],
    'min_sizes': [16, 32, 64, 128, 256, 512],
    'max_sizes': [],
    #'max_sizes': [8, 16, 32, 64, 128, 256],
    #'aspect_ratios': [ [],[],[],[],[],[] ],   # [1,2]  default 1
    'aspect_ratios': [ [1.5],[1.5],[1.5],[1.5],[1.5],[1.5] ],   # [1,2]  default 1
    
    'backbone': 'resnet152' , # vgg, resnet, detnet, resnet50
    'feature_pyramid_network':True ,
    'bottom_up_path': False ,
    'feature_enhance_module': True ,
    'max_in_out': True , 
    'focal_loss': False ,
    'progressive_anchor': True ,
    'refinedet': False ,   
    'max_out': False , 
    'anchor_compensation': False , 
    'data_anchor_sampling': False ,
   
    'overlap_thresh' : [0.4] ,
    'negpos_ratio':3 , 
    # test
    'nms_thresh':0.3 ,
    'conf_thresh':0.01 ,
    'num_thresh':5000 ,
}
