# Adapting NVIDIA's TF2 UNet model for AIcrowd's Mapping Challenge

## Data

Download the training dataset from the [crowdAI Mapping Challenge](https://www.aicrowd.com/challenges/mapping-challenge-old), and put the gzipped data archives into `data/raw/mapping-challenge/` folder.

```
|- data/
   |- raw/
      |- mapping-challenge
         |- test_images.tar.gz
         |- train.tar.gz
         |- val.tar.gz
```

## Set up Docker environment

Build the docker image.

```
$ docker build -t sat docker
```

Enter the docker container, passing through the project directory from the host. Also forward local port 8888 to the container to use Jupyter notebooks.

```
$ docker run --rm --gpus all -v ${PWD}:/code -v ${PWD}/data:/code/data -p 8888:8888 -it sat
```

Install the project directory mounted under `/code` in the container, as editable, so that we can make code changes to the python package sources and have them immediately auto-reloaded inside the container.

```
$ python -m pip install -e /code
```

## Prepare data

Convert the Mapping Challenge training dataset from MS-COCO format to TFRecord.

```
$ scripts/prepare_data.sh data
```

This produces TFRecords under the `processed` data directory:

```
|- data/
   |- raw/
      |- mapping-challenge/
         |- test_images.tar.gz
         |- train.tar.gz
         |- val.tar.gz
   |- processed/
      |- mapping-challenge/
         |- test_images/
            |- *.jpg
         |- train/
         |- val/
         |- train.record-*-of-00020
         |- val.record-*-of-00005
```

To run notebooks inside the container, install Jupyter and make sure to listen on all interfaces (0.0.0.0) so it can be reached from the outside.

```
$ python install -m pip install jupyterlab
$ jupyter lab --ip=0.0.0.0
```
