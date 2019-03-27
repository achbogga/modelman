from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import shutil
import argparse
import os
import sys
import glob
import pandas as pd
import numpy as np
from feature_extractor.batch import download_images
from activelearning.others.downloader import Downloader


def fetch_images(csv_path, image_path, n_workers=4):
    """This function is used to download all imgs from csv file
       if the image is not exists.

    Args:
      csv_path: the csv path
      img_path: the path to save downloaded images

    Returns:
        Nothing
    """

    csv_reference_df = pd.read_csv(csv_path,
                                   index_col=False,
                                   encoding='utf-8')
    image_url_list = csv_reference_df['ImgUrl'].unique().tolist()
    image_url_clean_list = []

    if not os.path.exists(image_path):
        os.makedirs(image_path)

    # Check the image is whether exist in the image pool
    for n in image_url_list:
        image_filename = n.split('/')[-1]
        temp_path = os.path.join(image_path, image_filename)

        if not os.path.isfile(temp_path):
            image_url_clean_list.append(n)

    # Download image if not in the image pool
    download_images(urls=image_url_clean_list,
                    n_workers=n_workers,
                    output_folder=image_path)


def prepare_images(args):
    output_images_dir = os.path.join(args.output_dir, 'all_images')
    output_cropped_images_dir = os.path.join(
        args.output_dir, 'all_cropped_images')

    # Locate .cvs and true labels for cropping
    filepath_list = [os.path.join(args.data_dir, n) for n in args.csv_file_list if os.path.exists(
        os.path.join(args.data_dir, n))]

    # Read head SKU list without reference images
    # sku_id_df = pd.read_csv(args.head_sku_list,
    #                        index_col=False, encoding='utf-8')
    # sku_id_df = sku_id_df.loc[:,0].tolist()
    # sku_id_df = [str(item) for item in sku_id_df]
    # print ('Debug1: ',sku_id_df[0:3])
    sum_df = pd.DataFrame()

    print(' -> Reading the head SKU list with no reference \
          images from: {}'.format(args.head_sku_list))

    for n in filepath_list:
        try:
            temp_df = pd.read_csv(n, index_col=False, encoding='utf-8')
        except IOError:
            print(n + ' File not found; skipping it!')
            continue
        temp_sku_ids = temp_df.SystemId.tolist()
        # print ('Debug2: ',temp_sku_ids[0:3])
        # temp_df = temp_df.loc[temp_df.SystemId.isin(sku_id_df)]
        # print (temp_df)
        sum_df = sum_df.append(temp_df)

        print(' -> Obtaining target .csv dataset from: {}'.format(n))

    # Save information to temp loaction
    sum_df.to_csv(args.sum_csv_name, index=False, encoding='utf-8')

    print(' -> The combination all .csv datasets and \
          save it into: {}'.format(args.sum_csv_name))

    # Begin downloading
    fetch_images(args.sum_csv_name, output_images_dir)

    # Uncomment blow code if you want clean saved all cropped images
    # if os.path.exists(output_cropped_images_dir):
    #    shutil.rmtree(output_cropped_images_dir)

    # Begin cropping
    Downloader.crop_save_sku(output_cropped_images_dir,
                             output_images_dir,
                             args.sum_csv_name,
                             ignore_ids=[-1, 0, 2, 1, 1265, 1000050])

    print('All done')


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Directory where all the csv files exist.',
                        default='/home/caffe/examples/sku_classification/')
    parser.add_argument('--output_dir', type=str, help='Output dir path for the generated_crops',
                        default='/home/caffe/images/ccna_train_data_v2_image_quality_impact')
    parser.add_argument('--csv_file_list', type=str, nargs='+', help='All the csv file names for the data', default=['test_data_sku_CCNA_20180619.csv', 'test_data_sku_CCNA_20180612.csv', 'test_data_sku_CCNA_20180614.csv',
                                                                                                                     'test_data_sku_CCNA_20180616.csv', 'test_data_sku_CCNA_20180623.csv', 'test_data_sku_CCNA_20180627.csv', 'test_data_sku_CCNA_20180629.csv', 'test_data_sku_CCNA_20180709.csv', 'test_data_sku_RCCB_20180817.csv'])
    parser.add_argument('--dataset_prefix', type=str,
                        help='The dataset prefix', default='ccna_train_data_v2_native_res')
    parser.add_argument('--head_sku_list', type=str, help='The delimiter that separates all the test image names with numbers',
                        default='/home/caffe/images/CCNA_head_sku_list.csv')
    parser.add_argument('--sum_csv_name', type=str, help='The summary csv file path',
                        default='/home/caffe/images/summary_ccna_train_data_v2_native_res.csv')
    return parser.parse_args(argv)


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    prepare_images(args)

# china_drinks_training_data = ['train_labels_20170522_washed.csv' 'train_labels_20170525_washed.csv' 'train_labels_20170527_washed.csv' 'train_labels_20170531_washed.csv' 'train_labels_20170613_washed.csv' 'train_labels_20170620_washed.csv' 'train_labels_20170626_washed.csv' 'train_labels_20170815_washed.csv' 'train_labels_20170823_washed.csv' 'train_labels_20170825_washed.csv' 'train_labels_20170928_washed.csv' 'train_shelves_labels_20170819_washed.csv' 'train_shelves_labels_20170826_washed.csv' 'train_xian_labels_20171012_washed.csv' 'train_beijing_labels_20171012_washed.csv' 'train_beijingxian_labels_20171110_coke_washed.csv' 'train_beijingxian_labels_20171110_noncoke_washed.csv' 'train_blf_full_20170913_washed.csv' 'train_massive_labels_20171221_washed.csv' 'train_nielsen_labels_20180105_washed.csv' 'train_cntoge_labels_20180105_washed.csv' 'train_massive_labels_20180108_washed.csv' 'train_cnt2336_labels_20180213_washed.csv' 'train_rb_labels_20180213_washed.csv' 'train_cnt24_labels_20180215_washed.csv' 'data_sku_KOcomp_20180215_washed.csv' 'train_sku_KOcomp_20180322_washed.csv' 'train_sku_nielsen_drinks_20180322_washed.csv' 'train_sku_cntoge_20180331_washed.csv' 'train_sku_pack_20180409_washed.csv' 'train_sku_pack_20180414_washed.csv' 'train_sku_rare_20180425_washed.csv' 'train_rare_sku_SwireBev_20180525_washed.csv' 'train_sku_pack_20180726_washed.csv' 'train_rare_sku_20180828.csv' 'train_rare_sku_20180903.csv' 'train_labnew_180907.csv' 'train_data_sku_INE_20180917.csv' 'train_data_sku_INE_20180922.csv' 'train_data_sku_INE_20180929.csv' 'train_data_sku_INE_20180930.csv' 'train_data_sku_INE_20181010.csv' 'train_data_sku_china_drink_20181126.csv' 'train_data_sku_china_drink_20181125.csv' 'train_data_sku_china_drink_20181130.csv' 'train_data_sku_china_drink_20190101_qa.csv']

# china_drinks_head_sku_list = '/home/caffe/caffe/examples/retail/sku_labels.csv'

# start = time.time()
# prepare_images(filename_list=china_drinks_training_data, INPUT_DIR = '/home/caffe/caffe/examples/sku_classification/', OUTPUT_DIR = '/home/caffe/images/china_drinks_image_resolution_exp_test_data', dataset_prefix = 'china_drinks_train', csv_filename_in = china_drinks_head_sku_list)
# print ('Total time taken to download all the china drinks training data: '+str(time.time()-start))
