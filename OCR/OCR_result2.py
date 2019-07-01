#!/usr/bin/env python

import io
import os
import sys
import time

import tensorflow as tf

#Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Default paths.
HANGUL_FILE = './Commercial_Hangul_2350.txt'
GRAPH_FILE = '../model/model6/optimized_hangul_recognition6.pb'
#GRAPH_FILE = './optimized_hangul_recognition.pb'
TEST_IMAGE = "../test_images"
#TEST_IMAGE = "./test_images"


#이미지 파일 읽어오는 함수
def read_image(file):
    #file_content에는 텐서플로우, 파일 읽어오기 함수로 읽어온 파일
    file_content = tf.read_file(file)
    # image에 불러온 파일을 1채널로 jpeg 확장자 디코딩
    image = tf.image.decode_jpeg(file_content, channels=1)
    # 위의 이미지를 32바이트 형식으로 변환
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # 위의 이미지를 64*64로 데이터 변경 없이 재배열
    image = tf.reshape(image, [64 * 64])

    #이미지 반환
    return image


#분류
def hangul_classify(test_image, labels, graph_file):
    #라벨 파일 연다
    labels = io.open(labels, 'r', encoding='utf-8').read().splitlines()

    #이미지를 찾을 수 없을 때
    for img in test_image:
        if not os.path.isfile(img):
            print('Error: %s not found.' % img)
            sys.exit(1)

    #그래프 불러오기
    with tf.gfile.GFile(graph_file, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    #그래프 설정
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name='hangul-model',
            producer_op_list=None
        )

    #노드 값들 불러오기
    x = graph.get_tensor_by_name('hangul-model/input:0')
    y = graph.get_tensor_by_name('hangul-model/output:0')
    keep_prob = graph.get_tensor_by_name('hangul-model/keep_prob:0')

    #이미지 읽어오기
    images = []
    for img in test_image:
        images.append(read_image(img))

    #세션 생성 및 시작
    sess = tf.InteractiveSession()
    image_array = sess.run(images)
    sess.close()
    with tf.Session(graph=graph) as graph_sess:
        result= ''
        for image in image_array:
            predictions = graph_sess.run(y, feed_dict={x: image, keep_prob: 1.0})
            sorted_index = predictions[0].argsort()[::-1][:1][0]
            result += labels[sorted_index]


    return result



if __name__ == '__main__':
    image_file = []
    image_file = list(os.walk(TEST_IMAGE))

    images = []
    for img in image_file[0][2]:
        images.append(image_file[0][0] + "/" + img)

    """
    # 시간 측정
    start_time = time.time()
    sh_start_time = time.strftime("[%Y%m%d %X]", time.localtime())
    """
    letters = hangul_classify(images, HANGUL_FILE, GRAPH_FILE)

    print(letters)
    """
    end_time = time.time()
    sh_end_time = time.strftime("[%Y%m%d %X]", time.localtime())
    print("%s ~ %s, Running Time: %s second" % (sh_start_time, sh_end_time, end_time - start_time))
    """
