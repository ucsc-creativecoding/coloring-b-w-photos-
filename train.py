import tensorflow as tf
import numpy as np


from utils import *
from model import *


def train(train_noisy_images, train_gt_images, val_noisy_images, val_gt_images):
    tf.reset_default_graph()

    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

    training_noisy_data = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNEL_COLOR])
    training_gt_data = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT, IMAGE_WIDTH, N_CHANNEL_COLOR])
    
    output = neural_network_with_skip_connections(training_noisy_data)

    loss = tf.reduce_mean(tf.losses.absolute_difference(labels=training_gt_data, predictions=output))
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        saver = initialize(sess)

        initial_step = global_step.eval()
    
        for index in range(initial_step, N_ITERATION):
            # print("Iteration "+ str(index))
            random_indices = get_random_indices(train_noisy_images.shape[0], BATCH_SIZE)
            
            noisy_batch = np.take(train_noisy_images, random_indices, 0)
            gt_batch = np.take(train_gt_images, random_indices, 0)

            sess.run(optimizer, feed_dict={training_noisy_data: noisy_batch, training_gt_data: gt_batch})

            if (index+1) % 100 == 0:
                saver.save(sess, CKPT_DIR, index)
                train_loss = sess.run(loss, feed_dict={training_noisy_data: noisy_batch, training_gt_data: gt_batch})
                val_loss, val_output = sess.run([loss, output], feed_dict={training_noisy_data: val_noisy_images, training_gt_data: val_gt_images})
                print("Iteration %d, train loss %g %%, validation loss %g %%"%(index+1, train_loss, val_loss))
                for i, (noisy_image, val_output_image) in enumerate(zip(val_noisy_images, val_output)):
                    scipy.misc.imsave("validation_results/"+str(i)+"_input_gray.jpg", noisy_image)
                    scipy.misc.imsave("validation_results/"+str(i)+"_output_color.jpg", val_output_image)


if __name__ == "__main__":
    train_gray_images = load_data_color(TRAIN_GRAY_DATASET_PATH)
    train_color_images = load_data_color(TRAIN_COLOR_DATASET_PATH)

    val_gray_images = load_data_color(VAL_GRAY_DATASET_PATH)
    val_color_images = load_data_color(VAL_COLOR_DATASET_PATH)

    print(train_gray_images.shape)
    print(train_color_images.shape)

    print(val_gray_images.shape)
    print(val_color_images.shape)


    train(train_gray_images, train_color_images, val_gray_images, val_color_images)
    
