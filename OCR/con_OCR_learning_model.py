import io
import sys, os
import time
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

from cnn_model import Model


#Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#Default paths.
HANGUL_FILE = "./Commercial_Hangul_2350.txt"
TFRECORDS_DIR = "../data/tfrecords/tfrecord6"
OUTPUT_DIR = "../model/model6"


MODEL_NAME = 'hangul_recognition6'
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
#752 > 73세대 : 55000 > / 658 > 15세대 : 10000 / 846 > 15세대: 13,000 > 17세대:15,000 / 470 > 19세대:9000
EPOCH = 9000
BATCH_SIZE = 100

LOAD_MODEL_STRUCT = "../model/model5/hangul_recognition5.chkp.meta"
LOAD_MODEL_VAL = "../model/model5/hangul_recognition5.chkp"


def load_model(model_struct):

    saver = tf.train.import_meta_graph(model_struct)


# 학습 데이터 가져오기(이미지 파일, 해당 이미지의 라벨)
def get_learning_data(files, num_labels):
    # 파일 큐,

    # 이미지 받아오기
    file_queue = tf.train.string_input_producer(files)

    # print(type(file_queues))#FIFOQueue
    # file_queue = tf.data.Dataset.from_tensor_slices(files)
    # file_queue = tf.data.TFRecordDataset(files)
    # features_data = tf.train.Feature(file_queue)
    # print(file_queue)
    # print(type(file_queue))

    # 학습 데이터는 tfrecords 파일 형식으로 되어 있다.
    reader = tf.TFRecordReader()
    _, value = reader.read(file_queue)

    # feature 정의, 각 지정된 타입으로 데이터를 읽는다.
    keys_to_features = {
        'image/class/label': tf.io.FixedLenFeature([], tf.int64, default_value=0),
        'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value='')
    }
    # parse_single_example로 feature 파싱, 딕셔너리 형태로 저장됨.
    features = tf.parse_single_example(value, features=keys_to_features)

    # print(features)

    label = features['image/class/label']
    image_encoded = features['image/encoded']

    # JPEG로 디코딩. 흑백 이미지니까 1채널
    image = tf.image.decode_jpeg(image_encoded, channels=1)
    # tf_float32 형태의 행렬로 변형
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # 행렬 데이터를 미리 정해둔 이미지 크기로 변형
    image = tf.reshape(image, [IMAGE_WIDTH * IMAGE_HEIGHT])

    # 라벨은 원-핫 인코딩 방식으로 변형
    label = tf.stack(tf.one_hot(label, num_labels))
    return image, label


def export_model(model_output_dir, input_node_names, output_node_name):

    name_base = os.path.join(model_output_dir, MODEL_NAME)
    input_graph_path = os.path.join(model_output_dir, MODEL_NAME + '.pbtxt')
    checkpoint_path = os.path.join(model_output_dir, './' + MODEL_NAME  + '.chkp')
    input_saver_def_path = ""
    input_binary = False
    restore_op_name = 'save/restore_all'
    filename_tensor_name = 'save/Const:0'
    clear_devices = True
    frozen_graph_file = os.path.join(model_output_dir, 'frozen_' + MODEL_NAME + '.pb')


    freeze_graph.freeze_graph( input_graph_path, input_saver_def_path, input_binary, checkpoint_path,
        output_node_name, restore_op_name, filename_tensor_name, frozen_graph_file, clear_devices, "")


    input_graph_def = tf.GraphDef()
    print(input_graph_def)
    with tf.gfile.Open(frozen_graph_file, "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def,
            input_node_names,
            [output_node_name],
            tf.float32.as_datatype_enum)

    optimized_graph_file = os.path.join(model_output_dir, 'optimized_' + MODEL_NAME + '.pb')
    f =  tf.gfile.GFile(optimized_graph_file, "wb")
    f.write(output_graph_def.SerializeToString())

    print("Inference optimized graph saved at: " + optimized_graph_file)



def learn_model(model_struct, model_val, label_file, tfrecords_dir, output_dir, n_epoch):
    # 라벨 읽어오기
    labels = io.open(label_file, 'r', encoding='utf-8').read().splitlines()
    num_labels = len(labels)

    # 노드 이름 지정
    input_node_name = 'input'
    keep_prob_node_name = 'keep_prob'
    output_node_name = 'output'


    print('Learning Start!')
    # 시간 측정
    start_time = time.time()
    sh_start_time = time.strftime("[%Y%m%d %X]", time.localtime())

    # 학습률, 테스팅률, 시간 기록
    log_file = open("learning_log6.txt", "w")

    # 학습 데이터
    # tfrecord 폴더에서, train-학습 데이터들 표기-으로 시작하는 파일들
    tf_record_pattern = os.path.join(tfrecords_dir, '%s-*' % 'train')
    # 위의 패턴으로 된 파일 목록들
    train_data_files = tf.gfile.Glob(tf_record_pattern)
    # 각 파일들에서 데이터 뽑기
    image, label = get_learning_data(train_data_files, num_labels)

    # 시험 데이터(위와 동일)
    tf_record_pattern = os.path.join(tfrecords_dir, '%s-*' % 'test')
    test_data_files = tf.gfile.Glob(tf_record_pattern)
    test_image, test_label = get_learning_data(test_data_files, num_labels)

    # 이미지, 라벨 묶음에서 랜덤으로 배치크기 만큼의 배치 객체 선정
    image_batch, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=BATCH_SIZE,
        capacity=2000,
        min_after_dequeue=1000)

    # 테스팅 데이터 배치. 위와 동일
    test_image_batch, test_label_batch = tf.train.batch(
        [test_image, test_label],
        batch_size=BATCH_SIZE,
        capacity=2000)



    # 플레이스 홀더 노드 지정(값 전달 노드)
    x = tf.placeholder(tf.float32, [None, IMAGE_WIDTH * IMAGE_HEIGHT], name=input_node_name)
    y_ = tf.placeholder(tf.float32, [None, num_labels])

    # 데이터 크기 맞추기
    x_image = tf.reshape(x, [-1, IMAGE_WIDTH, IMAGE_HEIGHT, 1])

    # 모델
    keep_prob = tf.placeholder(tf.float32, name=keep_prob_node_name)
    # 모델은 y
    y, y_pred = Model.CNN_model(x_image, keep_prob, num_labels, output_node_name)





    # 교차 엔트로피(비용함수) 정의
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=tf.stop_gradient(y_),
            logits=y
        )
    )

    # 아담옵티마이저 사용 > 경사하강법을 이용하여 학습
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    # 정확도 정의
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) #에러
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # 초기화 노드
    init = tf.global_variables_initializer()

    # 저장할 Saver() 노드 추가
    saver = tf.train.Saver()



    saver = tf.train.import_meta_graph(model_struct)

    with tf.Session() as sess:
        saver.restore(sess, model_val)


        # 학습 데이터 큐 쓰레드 초기화
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        # 체크 포인트 파일 저장 시의 디렉토리, 이름 설정
        checkpoint_file = os.path.join(output_dir, MODEL_NAME + '.chkp')

        # 파일 훈련 그래프 저장 시의 경로, 이름 설정
        tf.train.write_graph(sess.graph_def, output_dir, MODEL_NAME + '.pbtxt', True)


        plot_learn = []
        plot_loss = []
        plot_test = []

        # 학습(주어진 epoch 만큼)
        for epoch in range(n_epoch):
            # 이미지, 한글 라벨에서 임의의 배치를 가져온다
            train_images, train_labels = sess.run([image_batch, label_batch])
            # summary_str = acc_summary.eval(feed_dict = {x: train_images, y: train_labels})
            # file_writer.add_summary((summary_str, epoch))

            # 가져온 배치를 플레이스 홀더 노드 x, y_에 넣어주고 평가
            # keep_prob은 drop하지 않을 노드의 비율, scalar 텐서
            sess.run(train_step, feed_dict={x: train_images, y_: train_labels, keep_prob: 0.5})

            train_accuracy = sess.run(
                accuracy,
                feed_dict={x: train_images, y_: train_labels, keep_prob: 1.0}
            )
            plot_learn.append(float(train_accuracy) * 100)

            # if epoch % 100 == 0:
            #    print("Step %d, Training Accuracy %g" % (epoch, float(train_accuracy)))

            loss_print = cross_entropy.eval(
                feed_dict={x: train_images, y_: train_labels, keep_prob: 1.0})
            plot_loss.append(loss_print)

            if epoch % 100 == 0:
                # print("Step %d: Training Accuracy %g, Testing Accuracy %g" % (epoch, float(train_accuracy), accuracy_percent))
                # print("Step %d, Training Accuracy %g" % (epoch, float(train_accuracy)))
                print("Step %d, Training Accuracy %g, Training Loss %g" % (epoch, float(train_accuracy), loss_print))
                # print("Testing Accuracy {}".format(accuracy_percent))
                epoch_time = time.time()
                print("Step Running Time: %s second" % round(epoch_time - start_time, 4))
                log_file.write(
                    "Step %d, Training Accuracy %g, Training Loss %g" % (epoch, float(train_accuracy), loss_print))
                log_file.write(", Step Running Time: %s second\n\n" % round(epoch_time - start_time, 4))

                # 10,000번의 학습 때마다 체크포인트 파일 저장
                #3000 세대 마다
            if epoch % 3000 == 0:
                saver.save(sess, checkpoint_file, global_step=epoch)

        # 학습 종료 후 파일 저장
        saver.save(sess, checkpoint_file)

        # 테스트 파일 중 가져올 개수
        sample_count = 0
        for f in test_data_files:
            sample_count += sum(1 for _ in tf.python_io.tf_record_iterator(f))

        # 학습한 모델을 대상으로 테스팅 진행
        print('Testing Model!')

        # 배치 개수
        num_batches = int(sample_count / BATCH_SIZE) or 1
        total_correct_preds = 0

        # 테스트의 정확도
        test_accuracy = tf.reduce_sum(correct_prediction)
        # 배치 개수 만큼 테스팅
        for epoch in range(num_batches):
            # 테스트 배치를 가져온다
            t_image_batch, t_label_batch = sess.run([test_image_batch, test_label_batch])
            # accuracy2 노드에 플레이스 홀더 노드 x, y_에 각 테스트 배치 전달, 평가
            acc = sess.run(test_accuracy, feed_dict={x: t_image_batch,
                                                     y_: t_label_batch,
                                                     keep_prob: 1.0})
            total_correct_preds += acc

        # 정확도 계산, 출력
        accuracy_percent = total_correct_preds / (num_batches * BATCH_SIZE)
        print("Testing Accuracy {}".format(accuracy_percent))
        log_file.write("\n \nTesting Accuracy {}".format(accuracy_percent))
        plot_test = accuracy_percent * 100

        # 모델 저장
        export_model(output_dir, [input_node_name, keep_prob_node_name], output_node_name)

        end_time = time.time()
        sh_end_time = time.strftime("[%Y%m%d %X]", time.localtime())
        print("%s ~ %s, Running Time: %s second" % (sh_start_time, sh_end_time, end_time - start_time))
        log_file.write("\n%s ~ %s, Running Time: %s second" % (sh_start_time, sh_end_time, end_time - start_time))

        log_file.close()

        # plt.ylim(0.0, 1.2)
        # plt.ylim(0.0, 100.0)
        plt.plot(range(0, n_epoch), plot_learn, 'g', label='learning')
        plt.plot(range(0, n_epoch), plot_loss, 'r', label='loss')
        # plt.plot(range(0, n_epoch), train_loss_list, 'b', label='learning')
        plt.plot(range(n_epoch - 1, n_epoch), plot_test, 'or', label="Testing")
        # plt.plot(range(0, n_epoch), plot_test, 'r', label= "Testing")
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('OCR CNN model accuracy')
        plt.legend(loc='upper left')
        plt.show()

        # 학습 데이터 큐 쓰레드 닫기, 세션 종료
        coord.request_stop()
        coord.join(threads)
        sess.close()



if __name__ == '__main__':
    learn_model(LOAD_MODEL_STRUCT, LOAD_MODEL_VAL, HANGUL_FILE, TFRECORDS_DIR, OUTPUT_DIR, EPOCH)


