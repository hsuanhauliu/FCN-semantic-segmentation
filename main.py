import util
import numpy as np
import tensorflow as tf


def main():
    """ Main program """
    # Parameters
    train_data_size = 199
    batch_size = 1
    learning_rate = 0.001
    momentum = 0.99
    epochs = 2000
    show_image_step = train_data_size

    train_images_name = 'input_data/image/train/'
    train_labels_name = 'input_data/label/train/'
    train_gt_name = 'input_data/gt/train/'
    test_images_name = 'input_data/image/test/'
    test_labels_name = 'input_data/label/test/'
    test_gt_name = 'input_data/gt/test/'

    # read images and their labels
    training_images, training_labels, training_gt = util.read_data(train_images_name, train_labels_name, train_gt_name)
    test_images, test_labels, test_gt = util.read_data(test_images_name, test_labels_name, test_gt_name)

    # Split the training dataset into train and validation set
    train_images, vali_images = util.split_train_and_validation(training_images, train_data_size)
    train_labels, vali_labels = util.split_train_and_validation(training_labels, train_data_size)
    train_gt, vali_gt = util.split_train_and_validation(training_gt, train_data_size)

    vali_data_size = len(vali_images)
    test_data_size = len(test_images)

    train_labels = np.expand_dims(train_labels, axis=3)
    vali_labels = np.expand_dims(vali_labels, axis=3)
    test_labels = np.expand_dims(test_labels, axis=3)


    ############################# Graph Section #############################

    # Construct the graph
    print('Constructing the network...')

    # Inputs
    x = tf.placeholder(tf.float32, shape=(batch_size, 352, 1216, 3))
    y = tf.placeholder(tf.float32, shape=(batch_size, 352, 1216, 1))

    # Calculate Logits
    logits = util.fcn_32(x)

    # Mask: create a mask to ignore the void labels (-1)
    mask = tf.where(tf.not_equal(y, -1))
    logits_masked = tf.gather_nd(logits, mask)
    y_masked = tf.gather_nd(y, mask)

    # Optimization: calculate cross entropy -> get average loss -> call optimizer
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_masked, labels=y_masked)
    loss_op = tf.reduce_mean(cross_entropy)
    train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum).minimize(loss_op)

    # Prediction: calculate probabilities -> map road to true, non-road to false -> map to different colors
    y_predict_prob = tf.sigmoid(logits)
    y_predict_bool = tf.greater(y_predict_prob, 0.5)

    # Accuracy: filter out void labels using mask -> calculate IoU
    y_predict_masked = tf.gather_nd(y_predict_prob, mask)
    iou = util.calculate_iou(y_predict_masked, y_masked)

    # Record Loss summary
    loss_summary = tf.summary.scalar('Training Loss', loss_op)
    iou_summary = tf.summary.scalar('Training IoU', iou)

    mean_iou = tf.placeholder(tf.float32)
    mean_iou_summary = tf.summary.scalar('Mean IoU', mean_iou)

    # Training
    print('Start training...')
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter('logs' + '/train', sess.graph)
        vali_writer = tf.summary.FileWriter('logs' + '/vali', sess.graph)
        step_count = 0
        validation_count = 0
        test_count = 0

        while step_count < epochs:
            step_count += 1
            random_index = util.generate_random_index(train_data_size)
            image_batch = train_images[random_index:random_index + 1]
            label_batch = train_labels[random_index:random_index + 1]

            sess.run(train_op, feed_dict={x: image_batch, y: label_batch})
            l, loss, train_iou = sess.run([logits, loss_op, iou], feed_dict={x: image_batch, y: label_batch})

            # Record our results
            print('Step {}, Training data Loss: {}, Training data IoU: {}'.format(step_count, loss, train_iou))
            print('Max: {}, Min: {}'.format(np.max(l), np.min(l)))

            train_loss_summary, train_iou_summary = sess.run(
                [loss_summary, iou_summary],
                feed_dict={x: image_batch, y: label_batch})
            write_summaries(train_writer, step_count, train_loss_summary, train_iou_summary)

            # Every once in a while, do a prediction on the current train image and calculate validation mean IoU
            if step_count % show_image_step == 0:
                gt_batch = train_gt[random_index]
                util.output_image(image_batch[0],
                                 sess.run(y_predict_bool, feed_dict={x: image_batch, y: label_batch}),
                                 gt_batch, 'train', step_count)

                print('Start validating...')
                validation_count += 1
                total_vali_iou = 0
                for i in range(vali_data_size):
                    total_vali_iou += sess.run(iou, feed_dict={x: vali_images[i:i+1], y: vali_labels[i:i+1]})
                mean_vali_iou = total_vali_iou / vali_data_size
                mean_vali_iou_summary = sess.run(mean_iou_summary, feed_dict={mean_iou: mean_vali_iou})
                write_summaries(vali_writer, step_count, mean_vali_iou_summary)
                print('Step {}, Validation data mean IoU: {}'.format(step_count, mean_vali_iou))

            if step_count % 1000 == 0:
                # Testing: Calculate test data mean IoU
                print('Start testing...')
                test_count += 1
                total_test_iou = 0
                for i in range(test_data_size):
                    total_test_iou += sess.run(iou, feed_dict={x: test_images[i:i + 1], y: test_labels[i:i + 1]})
                    util.output_image(test_images[i],
                                     sess.run(y_predict_bool,
                                              feed_dict={x: test_images[i:i + 1], y: test_labels[i:i + 1]}),
                                     test_gt[i], 'test{}'.format(test_count), i)
                mean_test_iou = total_test_iou / test_data_size

                with open('results/test-result.txt', 'w') as wFile:
                    wFile.write('Step {}, Test data mean IoU: {}'.format(step_count, mean_test_iou))
        # Finish training
        train_writer.close()
        vali_writer.close()
    return


def write_summaries(writer, count, *argv):
    """ Add summaries to Tensorboard """
    for summary in argv:
        writer.add_summary(summary, count)


if __name__ == '__main__':
    main()
