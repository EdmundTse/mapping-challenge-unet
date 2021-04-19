#!/usr/bin/env bash

# Use first parameter if supplied, otherwise default to "data"
DATA_DIR=${1:-data}

# Prepare mapping-challenge dataset
TRAIN_GZ=$DATA_DIR/raw/mapping-challenge/train.tar.gz
VAL_GZ=$DATA_DIR/raw/mapping-challenge/val.tar.gz
TEST_GZ=$DATA_DIR/raw/mapping-challenge/test_images.tar.gz
PROCESSED_MAPPING_CHALLENGE=$DATA_DIR/processed/mapping-challenge
TRAIN_DIR=$PROCESSED_MAPPING_CHALLENGE/train
VAL_DIR=$PROCESSED_MAPPING_CHALLENGE/val
TESTIMG_DIR=$PROCESSED_MAPPING_CHALLENGE/test_images

# Extract the dataset .gz files
if [ ! -d $TRAIN_DIR ]; then
    echo Extract $TRAIN_GZ
    mkdir -p $PROCESSED_MAPPING_CHALLENGE
    tar -xzf $TRAIN_GZ -C $PROCESSED_MAPPING_CHALLENGE
else
    echo Directory $TRAIN_DIR exists, skipping extraction from archive.
fi
if [ ! -d $VAL_DIR ]; then
    echo Extract $VAL_GZ
    mkdir -p $PROCESSED_MAPPING_CHALLENGE
    tar -xzf $VAL_GZ -C $PROCESSED_MAPPING_CHALLENGE
else
    echo Directory $VAL_DIR exists, skipping extraction from archive.
fi
if [ ! -d $TESTIMG_DIR ]; then
    echo Extract $TEST_GZ
    mkdir -p $PROCESSED_MAPPING_CHALLENGE
    tar -xzf $TEST_GZ -C $PROCESSED_MAPPING_CHALLENGE
else
    echo Directory $TESTIMG_DIR exists, skipping extraction from archive.
fi

# Convert COCO dataset format into TFRecords
if [ ! -r $PROCESSED_MAPPING_CHALLENGE/train-small.record-00000-of-00001 ]; then
    echo Convert mapping-challenge train annotation-small.json to TFRecord
    python scripts/convert_coco_tfrecord.py \
        --annotations_file $TRAIN_DIR/annotation-small.json \
        --image_dir $TRAIN_DIR/images \
        --output_path $PROCESSED_MAPPING_CHALLENGE/train-small.record \
        --include_masks \
        --num_shards 1
fi
if [ ! -r $PROCESSED_MAPPING_CHALLENGE/train.record-00000-of-00020 ]; then
    echo Convert mapping-challenge train annotation.json to TFRecord
    python scripts/convert_coco_tfrecord.py \
        --annotations_file $TRAIN_DIR/annotation.json \
        --image_dir $TRAIN_DIR/images \
        --output_path $PROCESSED_MAPPING_CHALLENGE/train.record \
        --include_masks \
        --num_shards 20
fi
if [ ! -r $PROCESSED_MAPPING_CHALLENGE/val-small.record-00000-of-00001 ]; then
    echo Convert mapping-challenge val annotation-small.json to TFRecord
    python scripts/convert_coco_tfrecord.py \
        --annotations_file $VAL_DIR/annotation-small.json \
        --image_dir $VAL_DIR/images \
        --output_path $PROCESSED_MAPPING_CHALLENGE/val-small.record \
        --include_masks \
        --num_shards 1
fi
if [ ! -r $PROCESSED_MAPPING_CHALLENGE/val.record-00000-of-00005 ]; then
    echo Convert mapping-challenge val annotation.json to TFRecord
    python scripts/convert_coco_tfrecord.py \
        --annotations_file $VAL_DIR/annotation.json \
        --image_dir $VAL_DIR/images \
        --output_path $PROCESSED_MAPPING_CHALLENGE/val.record \
        --include_masks \
        --num_shards 5
fi
