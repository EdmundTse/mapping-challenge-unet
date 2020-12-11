# satellite

## Data

Download the training dataset from the [crowdAI Mapping Challenge](https://www.aicrowd.com/challenges/mapping-challenge-old), and put the `train.tar.gz` archive into `data/raw/mapping-challenge/` folder.

```
|- data/
   |- raw/
      |- mapping-challenge
         |- train.tar.gz
```

## Set up Docker environment

Build the docker image.

```
$ docker build -t sat docker
```

Enter the docker container, passing through the project directory from the host:

```
$ docker run --rm --gpus all -v ${PWD}:/code -v ${PWD}/data:/code/data -v /mnt/storage/new_mapping-challenge:/mnt/storage/new_mapping-challenge -it sat
```

Install the project directory mounted under `/code` in the container.

```
$ python -m pip install -e /code
```

## Prepare data

Convert the Mapping Challenge training dataset from MS-COCO format to TFRecord.

```
$ scripts/prepare_data.sh data
```

This produces TFRecords under the processed data directory:

```
|- data/
   |- raw/
      |- mapping-challenge/
         |- train.tar.gz
   |- processed/
      |- mapping-challenge/
         |- train/
         ...
         |- train.record-00000-of-00100
         |- train.record-00001-of-00100
         ...
         |- train.record-00099-of-00100
```