import os
import cv2
import time
import numpy as np
import tensorflow as tf

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
###############################################################################
#- 定义识别函数


def arrayreset(array):
    a = array[:, 0:len(array[0] - 2):3]
    b = array[:, 1:len(array[0] - 2):3]
    c = array[:, 2:len(array[0] - 2):3]
    a = a[:, :, None]
    b = b[:, :, None]
    c = c[:, :, None]
    m = np.concatenate((a, b, c), axis=2)
    return m



def recognize(src_image):
    """
    MobileNetV2-SSDLite
    :param src_image: 输入视频流或图像
    :param pb_file_path: the model file path
    :return: 
    """
    
    with tf.Graph().as_default():
        output_graph_def = tf.GraphDef()


        #---------------------------------------
        # 打开 .pb 模型
        pb_file = "ssd300_pascal_07+12_epoch-86_loss-1.2568_val_loss-0.5428.pb"

        with open(pb_file, "rb") as f:
            output_graph_def.ParseFromString(f.read())
            tensors = tf.import_graph_def(output_graph_def, name="")
            print("tensors:",tensors)


        with tf.Session() as sess:
           # init = tf.global_variables_initializer()
           # sess.run(init)

            #---------------------------------------
            # 打开图中所有的操作
            op = sess.graph.get_operations()
            for i,m in enumerate(op):
                print('op{}:'.format(i),m.values())

            #---------------------------------------
            # 模型的的输入和输出名称

            #--------------------------------------
            # 遍历某目录下的图像

            input_x = sess.graph.get_tensor_by_name("input_1:0")
            #print("input_X:",input_x)
            output_tensor = sess.graph.get_tensor_by_name("ssd_decoded_predictions/loop_over_batch/TensorArrayStack/TensorArrayGatherV3:0")
            #print("Output:",output_tensor)
     

            #--------------------------------------
            # 计算时间, 持续加载同一张图像

            src_image = arrayreset(src_image)       
            #src_image = cv2.imread(src_image) 
            org_img = src_image.copy()

            img=cv2.resize(src_image,(300,300))
            img=img.astype(np.float32)
                  
            y_pred = sess.run([output_tensor], feed_dict={input_x:np.reshape(img,(1,300,300,3))})
                      
            confidence_threshold = 0.8

            y_pred_array = np.array(y_pred[0])

            y_pred_thresh = [y_pred_array[k][y_pred_array[k,:,1] > confidence_threshold] for k in range(y_pred_array.shape[0])]
            classes = ['background', 'tank']
            image_size = (300, 300, 3)

            for box in y_pred_thresh[0]:
                xmin = box[2] * org_img.shape[1] / image_size[0]
                ymin = box[3] * org_img.shape[0] / image_size[1]
                xmax = box[4] * org_img.shape[1] / image_size[1]
                ymax = box[5] * org_img.shape[0] / image_size[0]
                label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
                print("label", label)


       


def main():
    src_image = "./002394.jpg"
    pb_file = "ssd300_pascal_07+12_epoch-86_loss-1.2568_val_loss-0.5428.pb"
    recognize(src_image)



if __name__ == '__main__':
    main()




