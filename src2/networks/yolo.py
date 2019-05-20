#Standard imports
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Reshape, Conv2D, Input, Lambda

class Yolo():
    def __init__(self, nb_class, batch_size, anchors, object_scale, max_box_per_image, 
                 no_object_scale, coord_scale, class_scale, nb_box, grid_h, grid_w, 
                 warmup_batches):
        
        self.batch_size = batch_size
        self.anchors = anchors
        self.object_scale = object_scale
        self.no_object_scale = no_object_scale
        self.coord_scale = coord_scale
        self.class_scale = class_scale              
        self.nb_box   = int(len(anchors)/2)
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.warmup_batches = warmup_batches
        self.max_box_per_image = max_box_per_image
        self.class_wt = np.ones(nb_class, dtype='float32')
        
    def build(self, input_shape, feature_extractor,
            labels):
    
        labels   = list(labels)
        nb_class = len(labels)
        nb_box   = self.nb_box
    
        max_box_per_image = self.max_box_per_image
    
        # make the feature extractor layers
        input_image = Input(input_shape)
        self.true_boxes = Input(shape=(1, 1, 1, max_box_per_image , 4))
        features = feature_extractor.extract(input_image)
    
        # make the object detection layer
        output = Conv2D(nb_box * (4 + 1 + nb_class), (1,1), strides=(1,1),
                        padding='same', name='DetectionLayer',
                        kernel_initializer='lecun_normal')(features)
        output = Reshape((self.grid_h, self.grid_w, nb_box, 4 + 1 + nb_class))(output)
        output = Lambda(lambda args: args[0])([output, self.true_boxes])
    
        model = Model([input_image, self.true_boxes], output)
    
        # initialize the weights of the detection layer
        layer = model.layers[-4]
        weights = layer.get_weights()
    
        new_kernel = np.random.normal(size=weights[0].shape)/(self.grid_h*self.grid_w)
        new_bias   = np.random.normal(size=weights[1].shape)/(self.grid_h*self.grid_w)
    
        layer.set_weights([new_kernel, new_bias])
    
        # print a summary of the whole model
        model.summary()
    
        return model
    
    def loss(self, y_true, y_pred):
        
        batch_size = self.batch_size
        anchors = self.anchors
        object_scale = self.object_scale
        no_object_scale = self.no_object_scale
        coord_scale = self.coord_scale
        class_scale = self.class_scale              
        nb_box   = self.nb_box
        grid_h = self.grid_h
        grid_w = self.grid_w  
        warmup_batches = self.warmup_batches
        class_wt = self.class_wt
                
        mask_shape = tf.shape(y_true)[:4]    
    
#        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)))
        cell_x = tf.cast(tf.reshape(tf.tile(tf.range(grid_w), [grid_h]), (1, grid_h, grid_w, 1, 1)),'float32')
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))
    
        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [batch_size, 1, 1, nb_box, 1])
    
        coord_mask = tf.zeros(mask_shape)
        conf_mask  = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)
    
        seen = tf.Variable(0.)
#        total_recall = tf.Variable(0.)
    
        """
        Adjust prediction
        """
        ### adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
    
        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(anchors, [1,1,1,nb_box,2])
    
        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])
    
        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]
    
        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2] # relative position to the containing cell
    
        ### adjust w and h
        true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically
    
        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half
        true_maxes   = true_box_xy + true_wh_half
    
        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half
        pred_maxes   = pred_box_xy + pred_wh_half
    
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]
    
        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
    
        true_box_conf = iou_scores * y_true[..., 4]
    
        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)
    
        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * coord_scale
    
        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]
    
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
    
        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)
    
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half
    
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
    
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    
        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
    
        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.cast(best_ious < 0.6, "float32") * (1 - y_true[..., 4]) * no_object_scale
    
        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * object_scale
    
        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(class_wt, true_box_class) * class_scale
    
        """
        Warm-up training
        """
#        no_boxes_mask = tf.to_float(coord_mask < coord_scale/2.)
        no_boxes_mask = tf.cast(coord_mask < coord_scale/2.,"float32")
        seen = tf.assign_add(seen, 1.)
    
        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, warmup_batches+1),
                              lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                       true_box_wh + tf.ones_like(true_box_wh) * \
                                       np.reshape(anchors, [1,1,1,nb_box,2]) * \
                                       no_boxes_mask,
                                       tf.ones_like(coord_mask)],
                              lambda: [true_box_xy,
                                       true_box_wh,
                                       coord_mask])
    
        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.cast(coord_mask > 0.0,'float32'))
        nb_conf_box  = tf.reduce_sum(tf.cast(conf_mask  > 0.0,'float32'))
        nb_class_box = tf.reduce_sum(tf.cast(class_mask > 0.0,'float32'))
    
        loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
    
        loss = tf.cond(tf.less(seen, warmup_batches+1),
                      lambda: loss_xy + loss_wh + loss_conf + loss_class + 10,
                      lambda: loss_xy + loss_wh + loss_conf + loss_class)
    
# =============================================================================
#         if DEBUG:
#             nb_true_box = tf.reduce_sum(y_true[..., 4])
#             nb_pred_box = tf.reduce_sum(tf.cast(true_box_conf > 0.5, 'float32') * tf.cast(pred_box_conf > 0.3, 'float32'))
#     
#             current_recall = nb_pred_box/(nb_true_box + 1e-6)
#             total_recall = tf.assign_add(total_recall, current_recall)
#     
#             loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
#             loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
#             loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
#             loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
#             loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
#             loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
#             loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)
# =============================================================================
    
        return loss
     
 
                               
