# -*- coding: UTF-8 -*-
from datetime import datetime
import numpy as np

import os
import tensorflow as tf
from tensorflow.keras.models import load_model

from CNN.model import YMLModel

from captcha_setting import IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES, LATENT_DIM, ALL_CHAR_SET, MAX_CAPTCHA, BATCH_SIZE, \
    CNN_EPOCH, VAL_PERCENT


def main():
    path = "../GeneratorModels/my_model_cluster_8_1000_0001.h5"
    saved_model = load_model(path)

    model = YMLModel(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES, LATENT_DIM, ALL_CHAR_SET, MAX_CAPTCHA)
    print(model.summary())

    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
        metrics=[tf.keras.metrics.Accuracy()]
    )
    model.fit(model=saved_model, batch_size=BATCH_SIZE, val_percent=VAL_PERCENT, epochs=CNN_EPOCH)

    model.save()


# c0 = captcha_setting.ALL_CHAR_SET[
#     np.argmax(predict_label[0, 0:captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
# c1 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0,
#                                             captcha_setting.ALL_CHAR_SET_LEN:2 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
# c2 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0,
#                                             2 * captcha_setting.ALL_CHAR_SET_LEN:3 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
# c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0,
#                                             3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
#
# # c3 = captcha_setting.ALL_CHAR_SET[np.argmax(predict_label[0, 3 * captcha_setting.ALL_CHAR_SET_LEN:4 * captcha_setting.ALL_CHAR_SET_LEN].data.numpy())]
# predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
# # predict_label = '%s%s%s%s' % (c0, c1, c2, c3)
# true_label = one_hot_encoding.decode(labels.numpy()[0])
# total += 1  # labels.size(0)
# # save_image(vimage,'temp_result/'+str(i)+'.png')
# # print(predict_label.upper(),'>>>>>',true_label)
# if (predict_label.upper() == true_label.upper()):
#     correct += 1
# # try:
#
#
# print('Test Accuracy of the model on the %d test images: %f %%' % (total, 100 * correct / total))
# # except:
# # 	pass
# loss_list.append(loss_total)
# with open("loss_record_6.txt", "wb") as fp:  # Pickling
#     pickle.dump(loss_list, fp)
# stop = datetime.now()
# # try:
# if (correct / total > accuracy):
#     accuracy = correct / total
#     # torch.save(cnn.state_dict(), "./model_lake/"+model_name.replace('.','_'+str(ttest_num)+'.'))   #current is model.pickle
#     torch.save(cnn.state_dict(), "./model_lake/" + model_name)  # current is model.pickle
#     print('saved!!!!!!!!!!!!!!!!!!!!!!!')
# # except:
# #  pass
# print("epoch:", epoch, "step:", i, " time:<", stop - start, "> loss:", loss_total)
# accuracy_list.append(accuracy)
# print(sum(accuracy_list) / len(accuracy_list))
# # torch.save(cnn.state_dict(), "./"+model_name)   #current is model.pkl
# # print("save last model")
# print(accuracy_list)

if __name__ == '__main__':
    main()
