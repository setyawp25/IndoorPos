import os
import h5py
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn as tflearn

import core.lstm_model as model

flags = tf.flags

# model params
flags.DEFINE_integer('hidden_size',         640,    'size of LSTM internal state')
flags.DEFINE_integer('rnn_layers',          1,      'number of layers in the LSTM')
# flags.DEFINE_integer('num_unroll_steps',    16,      'number of timesteps to unroll for = number of words to process per process')
flags.DEFINE_integer('num_unroll_steps',    2,      'number of timesteps to unroll for = number of words to process per process')
# flags.DEFINE_integer('num_class',            3,     'number of classes')
flags.DEFINE_integer('crd_xy_num',           2,     'number of Coordinates to predict : x, y')
flags.DEFINE_integer('crd_z_num',            6,     'number of one hot encoding Coordinates to predict : z')

flags.DEFINE_integer  ('img_cols',            103,     'column number of input data')
flags.DEFINE_integer  ('img_rows',            2,     'row number of input data')  # 32*31 = 992 = RSSI feature columns
# flags.DEFINE_float  ('img_cols',            28,     'column number of input data')
# flags.DEFINE_float  ('img_rows',            16,     'row number of input data')  # 32*31 = 992 = RSSI feature columns
flags.DEFINE_float  ('dropout',             0.0,    'dropout. 0 = no dropout')
flags.DEFINE_float  ('lambda_loss_amount',  0.0015, 'lambda loss amount')

# optimization
flags.DEFINE_float  ('learning_rate',       0.001,  'starting learning rate')
flags.DEFINE_integer('trn_batch_size',        24,   'number of sequences to train on in parallel')
flags.DEFINE_integer('tst_batch_size',        24,   'number of sequences to train on in parallel')
flags.DEFINE_integer('max_epochs',          100,    'number of full passes through the training data')

# bookkeeping
flags.DEFINE_integer('print_every',    5,    'how often to print current loss')
flags.DEFINE_string('logs_path',    "logs/",    'logs for Tensorboard')
flags.DEFINE_string('model_name',    "lstm",    'logs for Tensorboard')

FLAGS = flags.FLAGS

event_dirname = "{model_name}-{layers}-{hidden_size}".format(model_name=FLAGS.model_name, layers=str(FLAGS.rnn_layers), hidden_size=str(FLAGS.hidden_size))

print("hidden_size = ", FLAGS.hidden_size)
print("rnn_layers = ", FLAGS.rnn_layers)
print("Event DirName = ", event_dirname)
if not os.path.exists(os.path.join(FLAGS.logs_path, event_dirname)):
    os.makedirs(os.path.join(FLAGS.logs_path, event_dirname))

def distance(a, b):
    assert a.shape[0] == b.shape[0]
    if(a.shape[0] > 1):
        # print("1. a-b = %s\n2. pow(a-b) = %s\n3. sum(pow(a-b)) = %s\n4. sqrt(sum(pow(a-b))) = %s" % (a-b, np.power(a-b, 2), np.sum(np.power(a-b, 2), axis=1), np.sqrt(np.sum(np.power(a-b, 2), axis=1   ))))
        return np.sqrt(np.sum(np.power(a-b, 2), axis=1))
    else:
        # print("1. a-b = %s\n2. pow(a-b) = %s\n3. sum(pow(a-b)) = %s\n4. sqrt(sum(pow(a-b))) = %s" % (a-b, np.power(a-b, 2), np.sum(np.power(a-b, 2)), np.sqrt(np.sum(np.power(a-b, 2)   ))))
        return [np.sqrt(np.sum(np.power(a-b, 2)))]


def load_train_test_data(path, h5py_fname):

    hf = h5py.File(os.path.join(path, h5py_fname), 'r')
    print("h5py keys = ", list(hf.keys()))

    X_train = np.array(hf.get('dataset_X_train'))
    X_test = np.array(hf.get('dataset_X_test'))
    Y_train = np.array(hf.get('dataset_Y_train'))
    Y_test = np.array(hf.get('dataset_Y_test'))

    return X_train, X_test, Y_train, Y_test

def load_feature_selections(path, h5py_fname, method):

    hf = h5py.File(os.path.join(path, h5py_fname), 'r')
    print("h5py keys = ", list(hf.keys()))

    feat_sel = np.array(hf.get(method))

    return feat_sel

def main():
    dataset_dir_processed = 'dataset_processed'
    h5py_fname = 'fingerprint_dataset_processed.h5'
    save_feat_sel_dir = 'feature_selection_results'
    results_dir = 'statistic_results'
    X_train, X_test, y_train, y_test = load_train_test_data(dataset_dir_processed, h5py_fname)
    rf_feat_sels = load_feature_selections(path = save_feat_sel_dir,
                                           h5py_fname = 'random_forest.h5',
                                           method='random_forest')
    print(rf_feat_sels.shape, rf_feat_sels)
    X_train = X_train[:, rf_feat_sels]
    X_test  = X_test[:, rf_feat_sels]

    floors = np.unique(y_train[:,2])
    print("floors = ", floors)

    y_train_xy = y_train[:, 0:2].copy()
    y_test_xy  = y_test[:, 0:2].copy()
    y_train_z  = y_train[:, 2].copy()
    y_test_z   = y_test[:, 2].copy()

    y_train_z_one_hot_enc = np.zeros((len(y_train_z), int(np.max(floors))+1))
    y_train_z_one_hot_enc[np.arange(len(y_train_z)), y_train_z.astype(int)] = 1
    y_test_z_one_hot_enc = np.zeros((len(y_test_z), int(np.max(floors))+1))
    y_test_z_one_hot_enc[np.arange(len(y_test_z)), y_test_z.astype(int)] = 1

    print('X_train.shape = %s\ty_train_xy.shape = %s\ty_train_z_one_hot_enc.shape = %s\nX_test.shape = %s\ty_test_xy.shape = %s\ty_test_z_one_hot_enc.shape = %s' %
         (X_train.shape, y_train_xy.shape, y_train_z_one_hot_enc.shape, X_test.shape, y_test_xy.shape, y_test_z_one_hot_enc.shape))
    # print('Unique RSS(array) numbers = %s' % (np.unique(X_train, axis=0).shape[0]))

    # utils.save_numpy_to_img(img_dir='dataset_img', numpy_arr=X_train, img_rows=FLAGS.img_rows, img_cols=FLAGS.img_cols)
    # return 0
    # X_train = X_train.reshape((-1, FLAGS.img_rows, FLAGS.img_cols))
    X_test = X_test.reshape((-1, FLAGS.img_rows, FLAGS.img_cols))
    print('X_test.shape = ', X_test.shape)
    # return 0

        # # Graph input/output
        # # x = tf.placeholder(tf.float32, [None, FLAGS.num_unroll_steps, input_size])
        # x = tf.placeholder(tf.float32, [None, FLAGS.num_unroll_steps, FLAGS.img_cols])  # input_size = dim(features)
        # y = tf.placeholder(tf.float32, [None, FLAGS.crd_xy_num])
        # _batch_size = tf.placeholder(tf.int32, [])
        #
        # # regression
        # lstm_model = model.LSTM(_input=x,
        #                 _batch_size=_batch_size,
        #                 dropout=FLAGS.dropout,
        #                 hidden_size=FLAGS.hidden_size,
        #                 num_rnn_layers=FLAGS.rnn_layers,
        #                 num_unroll_steps=FLAGS.num_unroll_steps)
        #
        # lstm_output = lstm_model.Bi_directional_LSTM()
        # # lstm_output = lstm_model.LSTM_RNN()
        # prediction, loss_xy = tflearn.models.linear_regression(lstm_output, y)
        # train_op = tf.contrib.layers.optimize_loss(
        #         loss_xy, tf.contrib.framework.get_global_step(),
        #         optimizer='Adam',
        #         learning_rate=FLAGS.learning_rate)

    # classification

    # Start training
    np.random.seed(123)
    training_start_time = time.time()
    # Perform Training steps with "batch_size" amount of example data at each loop
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
    # with tf.Graph().as_default(), tf.Session() as sess:
        with tf.variable_scope("Inf_Model"):       # Variable scope allows you to create new variables and to share already created ones while providing checks to not create or share by accident.
            train_z_model = model.Inference(is_training=True,
                                    dropout=FLAGS.dropout,
                                    num_class=FLAGS.crd_z_num,
                                    input_size=FLAGS.img_cols,
                                    hidden_size=FLAGS.hidden_size,
                                    num_rnn_layers=FLAGS.rnn_layers,
                                    num_unroll_steps=FLAGS.num_unroll_steps,
                                    learning_rate=FLAGS.learning_rate,
                                    batch_size=FLAGS.trn_batch_size)
            train_z_logits = train_z_model.inference_graph()
            train_z_correct_pred, train_z_loss_op = train_z_model.loss_graph(train_z_logits)
            train_z_op, train_z_accuracy, train_z_correct_pred = train_z_model.training_graph(train_z_correct_pred, train_z_loss_op)

        with tf.variable_scope("Reg_Model"):
            train_xy_model = model.LSTM(dropout=FLAGS.dropout,
                                    hidden_size=FLAGS.hidden_size,
                                    crd_xy_num=FLAGS.crd_xy_num,
                                    input_size=FLAGS.img_cols,
                                    num_rnn_layers=FLAGS.rnn_layers,
                                    num_unroll_steps=FLAGS.num_unroll_steps,
                                    learning_rate=FLAGS.learning_rate,
                                    batch_size=FLAGS.trn_batch_size,
                                    lambda_loss_amount=FLAGS.lambda_loss_amount)
            train_xy_output = train_xy_model.Bi_directional_LSTM_regression
            ()
            # train_xy_prediction, train_xy_loss_op = train_xy_model.loss_graph(train_xy_output)
            train_xy_loss_op = train_xy_model.loss_graph(train_xy_output)
            train_xy_op = train_xy_model.training_graph(train_xy_loss_op)

        init = tf.global_variables_initializer()

        # Create a summary to monitor cost tensor
        tf.summary.scalar("reg_loss", train_xy_loss_op)
        # Merge all summaries into a single op
        merged_summary_op = tf.summary.merge_all()

        # Run the initializer
        sess.run(init)
        # print sess.run(init)

        # op to write logs to Tensorboard
        summary_writer = tf.summary.FileWriter(os.path.join(FLAGS.logs_path, event_dirname), graph=tf.get_default_graph())

        shuffle_idx = np.arange(X_train.shape[0])

        Error_3D_Train_List = []
        Error_2D_Train_List = []
        Error_3D_Test_List = []
        Error_2D_Test_List = []
        Floor_Loss_z_Train = []
        Floor_Loss_z_Test = []
        for epoch in range(FLAGS.max_epochs):
            error_3D_train_list = []
            error_2D_train_list = []
            error_3D_test_list = []
            error_2D_test_list = []
            floor_z_train_list = []
            floor_z_test_list = []

            epoch_start_time = time.time()
            count = 0
            # x_train_reader.shuffle_arr_list()
            np.random.shuffle(shuffle_idx)
            # print("shuffle_idx = %s and len(shuffle_idx) = %s" % (shuffle_idx, shuffle_idx.shape))
            X_train_shuffle = X_train[shuffle_idx]
            y_train_shuffle = y_train[shuffle_idx]
            y_train_xy_shuffle = y_train_xy[shuffle_idx]
            y_train_z_shuffle = y_train_z_one_hot_enc[shuffle_idx]

            train_fdetect = 0
            # for x_batch, y_batch_xy in x_train_reader.iter_batches():
            for batch_id in range(0, y_train_xy_shuffle.shape[0], FLAGS.trn_batch_size):
                count += 1
                idx = batch_id
                x_batch    = X_train_shuffle[idx:idx+FLAGS.trn_batch_size, :]
                y_batch    = y_train_shuffle[idx:idx+FLAGS.trn_batch_size, :]
                y_batch_xy = y_train_xy_shuffle[idx:idx+FLAGS.trn_batch_size, :]
                y_batch_z  = y_train_z_shuffle[idx:idx+FLAGS.trn_batch_size, :]
                x_batch    = x_batch.reshape((-1, FLAGS.img_rows, FLAGS.img_cols))
                # print('x_batch.shape = %s \t y_batch_xy.shape = %s'% (x_batch.shape, y_batch_xy.shape))

                start_time = time.time()

                train_xy_pred, train_z_pred, train_loss_xy, train_loss_z, train_z_acc, train_xy_op_, train_z_op_, summary = \
                                sess.run([train_xy_output,
                                          train_z_correct_pred,
                                          train_xy_loss_op,
                                          train_z_loss_op,
                                          train_z_accuracy,
                                          train_xy_op,
                                          train_z_op,
                                          merged_summary_op],
                                feed_dict={train_z_model.input_:    x_batch,
                                           train_z_model.output_:   y_batch_z,
                                           train_xy_model.input_:   x_batch,
                                           train_xy_model.output_: y_batch_xy})
                floor_z_train_list.append(train_loss_z)

                # Write logs at every iteration
                summary_writer.add_summary(summary, epoch * y_train_xy_shuffle.shape[0] + batch_id)
                # print('pred = ', pred[0])]
                # y_batch_xy = np.squeeze(y_batch_xy)
                # pred = np.squeeze(pred)
                    # train_pred[0][2] = floors[np.argmin(np.abs(floors-round(train_pred[0][2], 1)))]
                xyz_pred = np.column_stack([train_xy_pred, train_z_pred])
                # print(train_xy_pred)
                # print(train_z_pred)
                # print('*'*50)
                # print(y_batch_xy)

                    # error_2D_train = distance(train_pred, y_batch_xy)
                    # avg_error_2D_train = np.mean(np.array(error_2D_train))
                error_3D_train = distance(xyz_pred, y_batch)
                avg_error_3D_train = np.mean(np.array(error_3D_train))

                # print('-'*50)
                # print('\t\t +++ Prediction = %s\n\t\t +++ GroundTruth = %s' % (train_pred, y_batch_xy))
                # print(error_3D_train)
                error_3D_train_list.extend(error_3D_train)
                # print('len of error_3D_train_list = ', len(error_3D_train_list))
                    # error_2D_train_list.extend(error_2D_train)
                error_2D_train_tmp = []
                for flr_idx in range(0, len(xyz_pred)):
                    # train_fdetect += round(xyz_test_pred[flr_idx][2], 1) == y_test_batch[flr_idx][2]
                    if round(xyz_pred[flr_idx][2], 1) == y_batch[flr_idx][2]:
                        train_fdetect += 1
                        # print(xyz_pred[flr_idx].shape)
                        # print(y_batch[flr_idx].shape)
                        error_2D_train = distance(xyz_pred[flr_idx].reshape(1, -1), y_batch[flr_idx].reshape(1, -1))
                        error_2D_train_tmp.append(error_2D_train)
                        error_2D_train_list.extend(error_2D_train)

                avg_error_2D_train = np.mean(np.array(error_2D_train_tmp))
                # print(error_3D_train_list)
                # print("train_loss_xy : Tensor train_loss_xy shape : ", np.array(train_loss_xy).shape)
                time_elapsed = time.time() - start_time
                if count % FLAGS.print_every == 0:
                    # print('%3d: [%5d/%5d], train_loss_xy = %.8f,\terror_3D_train = %s,\tsecs/batch = %.4fs,\n\t\tpred = %s --> ans = %s\n%s' % (epoch,
                    #                                         count,
                    #                                         X_train.shape[0],
                    #                                         train_loss_xy,
                    #                                         error_3D_train,
                    #                                         time_elapsed,
                    #                                         train_pred[0], y_batch_xy[0],
                    #                                         '-'*100))
                    print('%3d: [%5d/%5d], train_loss_xy = %.8f,\ttrain_loss_z = %.8f,\tavg_error_2D_train = %s,\tavg_error_3D_train = %s,\tsecs/batch = %.4fs\n%s' % (epoch,
                                                            count,
                                                            X_train.shape[0]/FLAGS.trn_batch_size,
                                                            train_loss_xy,
                                                            train_loss_z,
                                                            avg_error_2D_train,
                                                            avg_error_3D_train,
                                                            time_elapsed,
                                                            '+'*100))
                    # print('+++ CNN_output = %s with shape : %s\n+++ Prediction = %s with shape : %s\n+++ GroundTruth = %s with shape : %s' % \
                    #      (cnn_op, cnn_op.shape, train_pred, train_pred.shape, y_batch_xy, y_batch_xy.shape))
                    # print('+++ Prediction = %s with shape : %s\n+++ GroundTruth = %s with shape : %s' % \
                    #      (train_pred, train_pred.shape, y_batch_xy, y_batch_xy.shape))
                    # print('+'*50)

                    # print('\t\t%s' % ('-'*60))
                    # Evaluation on the test set (no learning made here - just evaluation for diagnosis)

            # print('+++ Mean positioning of training error (3D): \t%.3lf m' % np.mean(np.array(error_3D_train_list)))
            print('%s Epoch training time: %s %s' % ('='*10, time.time()-epoch_start_time, '='*10))
            # Calculate accuracy for testing data
            # print "\t> Testing Accuracy: %s" % (sess.run(accuracy, feed_dict={x: X_test, y: Y_test, batch_size: Y_test.shape[0]}))
            print('\"\"'*50)
            # Evaluation on the test set

            count = 0
            test_fdetect = 0
            for batch_id in range(0, y_test.shape[0], FLAGS.tst_batch_size):
                testing_start_time = time.time()
                count += 1
                idx = batch_id
                X_test_batch    = X_test[idx:idx+FLAGS.tst_batch_size, :]
                y_test_batch    = y_test[idx:idx+FLAGS.tst_batch_size, :]
                y_test_batch_xy = y_test_xy[idx:idx+FLAGS.tst_batch_size, :]
                y_test_batch_z  = y_test_z_one_hot_enc[idx:idx+FLAGS.tst_batch_size, :]

                test_xy_pred, test_z_pred, test_loss_xy, test_loss_z = \
                        sess.run([train_xy_output,
                                  train_z_correct_pred,
                                  train_xy_loss_op,
                                  train_z_loss_op],
                       feed_dict={train_xy_model.input_:    X_test_batch,
                                  train_xy_model.output_:   y_test_batch_xy,
                                  train_z_model.input_:     X_test_batch,
                                  train_z_model.output_:    y_test_batch_z})
                floor_z_test_list.append(test_loss_z)
                # test_pred = np.squeeze(test_pred)
                xyz_test_pred = np.column_stack([test_xy_pred, test_z_pred])
                    # test_pred[0][2] = floors[np.argmin(np.abs(floors-round(test_pred[0][2], 1)))]
                error_3D_test = distance(xyz_test_pred, y_test_batch)
                avg_error_3D_test = np.mean(np.array(error_3D_test))
                error_3D_test_list.extend(error_3D_test)
                # print('len of error_3D_test_list = ', len(error_3D_test_list))

                error_2D_test_tmp = []
                for flr_idx in range(0, len(xyz_test_pred)):
                    # test_fdetect += round(xyz_test_pred[flr_idx][2], 1) == y_test_batch[flr_idx][2]
                    if round(xyz_test_pred[flr_idx][2], 1) == y_test_batch[flr_idx][2]:
                        test_fdetect += 1
                        error_2D_test = distance(xyz_test_pred[flr_idx].reshape(1, -1), y_test_batch[flr_idx].reshape(1, -1))
                        error_2D_test_tmp.append(error_2D_test)
                        error_2D_test_list.extend(error_2D_test)

                avg_error_2D_test = np.mean(np.array(error_2D_test_tmp))


                        # for flr_idx in range(0, len(test_pred)):
                        #     fdetect += round(test_pred[flr_idx][2], 1) == y_test_batch[flr_idx][2]
                    # print(test_pred[flr_idx])
                # fdetect += round(test_pred[0][2], 1) == y_test_batch[0][2]
                # # if round(test_pred[0][2], 1) == y_test_batch[0][2]:
                # if np.abs(round(test_pred[0][2], 1) - y_test_batch[0][2]) <= 0.001 :    # cuz round(test_pred[0][2], 1) still gets ?.100000xxxx instead of ?.1
                #     error_2D_test = distance(test_pred, y_test_batch)
                #     error_2D_test_list.append(error_2D_test)
                # else:
                #     error_2D_test = "nan"
                time_elapsed = time.time() - start_time
                if count % ((FLAGS.print_every)*3) == 0:
                    # print('\t >>> [%5d/%5d], test_loss_xy = %s,\terror_3D_test = %s,\terror_2D_test = %s\n\t\tpred = %s --> ans = %s\n\t%s' %
                    #         (count,
                    #         y_test.shape[0],
                    #         test_loss_xy,
                    #         error_3D_test,
                    #         error_2D_test,
                    #         test_pred[0],
                    #         y_test_batch[0],
                    #         '-'*100))

                    print('\t >>> [%5d/%5d], test_loss_xy = %s,\ttest_loss_z = %s,\tavg_error_2D_test = %s\tavg_error_3D_test = %s,\tsecs/batch = %.4fs\n%s' %
                            (count,
                            y_test.shape[0]/FLAGS.tst_batch_size,
                            test_loss_xy,
                            test_loss_z,
                            avg_error_2D_test,
                            avg_error_3D_test,
                            time_elapsed,
                            '*'*100))
                    # print('+++ CNN_output = %s with shape : %s\n+++ Prediction = %s with shape : %s\n+++ GroundTruth = %s with shape : %s' % \
                    #      (test_cnn_op, test_cnn_op.shape, test_pred, test_pred.shape, y_test_batch, y_test_batch.shape))
                    # print('*** Prediction = %s with shape : %s\n*** GroundTruth = %s with shape : %s' % \
                    #      (test_pred, test_pred.shape, y_test_batch, y_test_batch.shape))
                    # print('*'*50)

            print('+++ Mean positioning of training error (3D): \t%.3lf m' % np.mean(np.array(error_3D_train_list)))
            print('+++ Mean positioning of training error (2D): \t%.3lf m' % np.mean(np.array(error_2D_train_list)))
            print('+++ (Train) Floor detection rate: \t\t%2.2lf %%' % ((float(train_fdetect) / y_train.shape[0])*100))
            print()
            print('*** Mean positioning of testing error (3D): \t%.3lf m' % np.mean(np.array(error_3D_test_list)))
            print('*** Mean positioning of testing error (2D): \t%.3lf m' % np.mean(np.array(error_2D_test_list)))
            print('*** (Test) Floor detection rate: \t\t%2.2lf %%' % ((float(test_fdetect) / y_test.shape[0])*100))
            print('%s Epoch testing time: %s %s' % ('='*10, time.time()-testing_start_time, '='*10))

            Error_2D_Train_List.append(np.mean(np.array(error_2D_train_list)))
            Error_3D_Train_List.append(np.mean(np.array(error_3D_train_list)))
            Error_2D_Test_List.append(np.mean(np.array(error_2D_test_list)))
            Error_3D_Test_List.append(np.mean(np.array(error_3D_test_list)))
            Floor_Loss_z_Train.append(np.mean(np.array(floor_z_train_list)))
            Floor_Loss_z_Test.append(np.mean(np.array(floor_z_test_list)))
        print("Optimization Finished!")

    print("Total training time = ", time.time()-training_start_time)
    # save numpy array to h5py
    stat_hf = h5py.File(os.path.join(results_dir, 'statistic_results_RMSE.h5'), 'w')
    stat_hf.create_dataset('Error_2D_Train_List', data=np.array(Error_2D_Train_List))
    stat_hf.create_dataset('Error_3D_Train_List', data=np.array(Error_3D_Train_List))
    stat_hf.create_dataset('Error_2D_Test_List', data=np.array(Error_2D_Test_List))
    stat_hf.create_dataset('Error_3D_Test_List', data=np.array(Error_3D_Test_List))
    stat_hf.create_dataset('Floor_Loss_z_Train', data=np.array(Floor_Loss_z_Train))
    stat_hf.create_dataset('Floor_Loss_z_Test', data=np.array(Floor_Loss_z_Test))
    stat_hf.close()
    print("%s Saving to h5py successfully %s" % ('='*15, '='*15))

if __name__ == '__main__':
    main()
