#!/usr/bin/env python
#참고: https://github.com/IBM/tensorflow-hangul-recognition

import glob
import io
import math
import os
import random

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


#폴더 경로
HANGUL_FILE = './Commercial_Hangul_2350.txt'
FONTS_DIR = '../data/fonts/font7'
OUTPUT_DIR = '../data'
HANGUL_LABEL = '../data/hangul_labels7.csv'
CSV_OUTPUT_DIR = '../data/tfrecords/tfrecord7'


DEFAULT_NUM_SHARDS_TRAIN = 3
DEFAULT_NUM_SHARDS_TEST = 1

#탄성변형 이미지 개수 지정
DISTORTION_COUNT = 3

#이미지 데이터의 높이, 넓이 지정
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

#shard
NUM_SHARDS_TRAIN = 3
NUM_SHARDS_TEST = 1

#ttf타입의 폰트 파일, 라벨과 매칭된 리스트 형태의 CSV 파일로 저장.
#한글 이미지 생성(한글 라벨 파일, 폰트 폴더, 결과 저장 폴더):
def create_images(label_file, fonts_dir, output_dir):

    #파일 열기(READ, UTF-8로 인코딩) 'f'이름으로
    with io.open(label_file, 'r', encoding='utf-8') as f:
        #f를 한 줄 단위로 labels에 넣는다.
        labels = f.read().splitlines()

    #파일 접근 > 결과 저장 폴더/hangul-images 폴더!
    image_dir = os.path.join(output_dir, 'hangul-images')
    #만약 위의 결과폴더/hangul-images폴더가 없으면
    if not os.path.exists(image_dir):
        #해당 폴더를 만든다.
        os.makedirs(os.path.join(image_dir))


    #폰트 폴더의 .ttf 타입 파일의 리스트를 반환, fonts에 저장한다.
    fonts = glob.glob(os.path.join(fonts_dir, '*.ttf'))

    #결과저장폴더/labels-map.csv를 WRITE, UTF-8인코딩 방식으로 열고, labels_csv라 한다.
    labels_csv = io.open(os.path.join(output_dir, 'hangul_labels7.csv'), 'w', encoding='utf-8')

    #변수 지정
    #훈련 데이터 개수
    total_count = 0
    prev_count = 0
    #labels(글자 리스트)의 글자 수 만큼 반복
    for character in labels:
        #만약 (글자 총 개수)-(진행한 개수)가 5000개 이상이면
        if total_count - prev_count > 5000:
            #
            prev_count = total_count
            print('{} images generated...'.format(total_count))

        #폰트 리스트의 폰트 개수 만큼 반복
        for font in fonts:
            #최종 개수에 1을 더한다.
            total_count += 1
            #image는 흑백 모드(L), 지정한 크기, 0의 바탕색
            image = Image.new('L', (IMAGE_WIDTH, IMAGE_HEIGHT), color=0)
            #폰트는 truetype(ttf)로, 반복문의 폰트, 48사이즈
            font = ImageFont.truetype(font, 48)
            #그리기는 위의 바탕을 먼저 그리고
            drawing = ImageDraw.Draw(image)
            #바탕의 반복문 문자를 반복문 폰트로 높이, 길이
            w, h = drawing.textsize(character, font=font)
            #바탕에 글자 삽입
            drawing.text(
                #(설정 글자 넓이-글자 원래 넓이)/2, (설정 글자 높이-글자 원래 높이)/2)로 크기를 설정,
                ((IMAGE_WIDTH-w)/2, (IMAGE_HEIGHT-h)/2),
                #글자는 위의 반복문 해당 글자
                character,
                #255로 채우고
                fill=(255),
                #글자체는 반복문의 해당 폰트
                font=font
            )
            #파일은 'hangul_전체 이미지 개수.jpeg'
            file_name = 'hangul_{}.jpeg'.format(total_count)
            #파일 경로는 위에서 받았던 image_dir/hangul_count.jpeg
            file_path = os.path.join(image_dir, file_name)
            #jpeg 파일로 저장
            image.save(file_path, 'JPEG')
            #파일에, 해당 문자를 매칭해서 csv에 쓴다.
            labels_csv.write(u'{},{}\n'.format(file_path, character))

            #탄성 변형 개수 지정된 만큼 반복
            for i in range(DISTORTION_COUNT):
                #전체 개수 +1
                total_count += 1
                #파일 이름은 hangul_전체 개수
                file_name = 'hangul_{}.jpeg'.format(total_count)
                #image_dir/hangul_ 경로 지정(파일명 지정)
                file_path = os.path.join(image_dir, file_name)
                #이미지 배열 생성, arr에 넣는다
                arr = np.array(image)

                #distorted_array에 이미지 배열, (30, 36)사이의 랜덤값을 알파, (5, 6)사이의 랜덤값을 시그마로 이미지 변형
                distorted_array = elastic_transform(arr, alpha=random.randint(30, 36), sigma=random.randint(5, 6))
                #위의 함수로 리턴된 변형 이미지 행렬을 이미지로 변경
                distorted_image = Image.fromarray(distorted_array)
                #이미지를 JPEG 확장자로 저장
                distorted_image.save(file_path, 'JPEG')
                #labels_csv에 매핑 정보(이미지 파일, 글자) 저장
                labels_csv.write(u'{},{}\n'.format(file_path, character))

    #이미지 변경 끝 출력
    print('Finished generating {} images.'.format(total_count))
    #매핑 파일 닫기
    labels_csv.close()


#이미지 탄성변형 시키기
#elastic_transform 함수 정의 (이미지, 알파값(변형 강도 조절 스케일값), 시그마값(가우시안 필터 표준 편차))
def elastic_transform(image, alpha, sigma, random_state=None):

    #랜덤값, 모두 같은 랜덤값을 사용하도록.
    if random_state is None:
        random_state = np.random.RandomState(None)
    #이미지 행렬 형태
    shape = image.shape

    #dx에 가우시안 필터(이미지 행렬의 랜덤값) * 2 -1, 시그마값, constant모드(엣지는 일정한 값으로 채운다))* 알파값
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    #dy에 가우시안 필터(이미지 행렬의 랜덤값) * 2 -1, 시그마값, constant모드(엣지는 일정한 값으로 채운다))* 알파값
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    #x, y에 이미지 행렬의 각 0번행, 1번 행 벡터에 대하여 각 사각형 영역을 이루는 조합을 입력
    #x, y = numpy.meshgrid(numpy.arange(shape[0]), numpy.arange(shape[1]))
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    #내부 데이터 보존해서 형태 바꿈.
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))
    #보간법으로 위에서 변형시킨 image 배열을 새 좌표에 매핑 후, 원래의 이미지 형태 행렬로 형태 변경
    return map_coordinates(image, indices, order=1).reshape(shape)


#if __name__ == '__main__':
    #create_images(HANGUL_FILE, FONTS_DIR, OUTPUT_DIR)



"""





"""
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class TFRecordsConverter(object):
    """Class that handles converting images to TFRecords."""

    def __init__(self, labels_csv, label_file, output_dir,
                 num_shards_train, num_shards_test):

        self.output_dir = output_dir
        self.num_shards_train = num_shards_train
        self.num_shards_test = num_shards_test

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Get lists of images and labels.
        self.filenames, self.labels = \
            self.process_image_labels(labels_csv, label_file)

        # Counter for total number of images processed.
        self.counter = 0

    def process_image_labels(self, labels_csv, label_file):
        """This will constuct two shuffled lists for images and labels.

        The index of each image in the images list will have the corresponding
        label at the same index in the labels list.
        """
        labels_csv = io.open(labels_csv, 'r', encoding='utf-8')
        labels_file = io.open(label_file, 'r',
                              encoding='utf-8').read().splitlines()

        # Map characters to indices.
        label_dict = {}
        count = 0
        for label in labels_file:
            label_dict[label] = count
            count += 1

        # Build the lists.
        images = []
        labels = []
        for row in labels_csv:
            file, label = row.strip().split(',')
            images.append(file)
            labels.append(label_dict[label])

        # Randomize the order of all the images/labels.
        shuffled_indices = list(range(len(images)))
        random.seed(12121)
        random.shuffle(shuffled_indices)
        file_names = [images[i] for i in shuffled_indices]
        labels = [labels[i] for i in shuffled_indices]

        return file_names, labels

    def write_tfrecords_file(self, output_path, indices):
        """Writes out TFRecords file."""
        writer = tf.python_io.TFRecordWriter(output_path)
        for i in indices:
            filename = self.filenames[i]
            label = self.labels[i]
            with tf.gfile.FastGFile(filename, 'rb') as f:
                im_data = f.read()

            # Example is a data format that contains a key-value store, where
            # each key maps to a Feature message. In this case, each Example
            # contains two features. One will be a ByteList for the raw image
            # data and the other will be an Int64List containing the index of
            # the corresponding label in the labels list from the file.
            example = tf.train.Example(features=tf.train.Features(feature={
                'image/class/label': _int64_feature(label),
                'image/encoded': _bytes_feature(tf.compat.as_bytes(im_data))}))
            writer.write(example.SerializeToString())
            self.counter += 1
            if not self.counter % 1000:
                print('Processed {} images...'.format(self.counter))
        writer.close()

    def convert(self):
        """This function will drive the conversion to TFRecords.

        Here, we partition the data into a training and testing set, then
        divide each data set into the specified number of TFRecords shards.
        """

        num_files_total = len(self.filenames)

        # Allocate about 15 percent of images to testing
        num_files_test = int(num_files_total * .15)

        # About 85 percent will be for training.
        num_files_train = num_files_total - num_files_test

        print('Processing training set TFRecords...')

        files_per_shard = int(math.ceil(num_files_train /
                                        self.num_shards_train))
        start = 0
        for i in range(1, self.num_shards_train):
            shard_path = os.path.join(self.output_dir,
                                      'train-{}.tfrecords'.format(str(i)))
            # Get a subset of indices to get only a subset of images/labels for
            # the current shard file.
            file_indices = np.arange(start, start+files_per_shard, dtype=int)
            start = start + files_per_shard
            self.write_tfrecords_file(shard_path, file_indices)

        # The remaining images will go in the final shard.
        file_indices = np.arange(start, num_files_train, dtype=int)
        final_shard_path = os.path.join(self.output_dir,
                                        'train-{}.tfrecords'.format(
                                            str(self.num_shards_train)))
        self.write_tfrecords_file(final_shard_path, file_indices)

        print('Processing testing set TFRecords...')

        files_per_shard = math.ceil(num_files_test / self.num_shards_test)
        start = num_files_train
        for i in range(1, self.num_shards_test):
            shard_path = os.path.join(self.output_dir,
                                      'test-{}.tfrecords'.format(str(i)))
            file_indices = np.arange(start, start+files_per_shard, dtype=int)
            start = start + files_per_shard
            self.write_tfrecords_file(shard_path, file_indices)

        # The remaining images will go in the final shard.
        file_indices = np.arange(start, num_files_total, dtype=int)
        final_shard_path = os.path.join(self.output_dir,
                                        'test-{}.tfrecords'.format(
                                            str(self.num_shards_test)))
        self.write_tfrecords_file(final_shard_path, file_indices)

        print('\nProcessed {} total images...'.format(self.counter))
        print('Number of training examples: {}'.format(num_files_train))
        print('Number of testing examples: {}'.format(num_files_test))
        print('TFRecords files saved to {}'.format(self.output_dir))


if __name__ == '__main__':
    create_images(HANGUL_FILE, FONTS_DIR, OUTPUT_DIR)
    converter = TFRecordsConverter(HANGUL_LABEL,
                                   HANGUL_FILE,
                                   CSV_OUTPUT_DIR,
                                   NUM_SHARDS_TRAIN,
                                   NUM_SHARDS_TEST)
    converter.convert()
