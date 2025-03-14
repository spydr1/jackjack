# Installation

A. Linux
1. Follow the official document.
```
https://www.tensorflow.org/install/source?hl=ko&_gl=1*1k1bqne*_up*MQ..*_ga*MjA1Njc0NzE3Mi4xNzQxNzcxNzU5*_ga_W0YLR4190T*MTc0MTc3MTc1OS4xLjAuMTc0MTc3MTc1OS4wLjAuMA..#docker_linux_builds
```

2. Run container
```
--GPU--
docker run --gpus all -it -w /tensorflow -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" tensorflow/tensorflow:2.16.1-gpu bash

--CPU--
docker run --gpus all -it -w /tensorflow -v $PWD:/mnt -e HOST_PERMS="$(id -u):$(id -g)" tensorflow/tensorflow:2.16.1 bash
```

3. In Container
```
apt update
apt-get install git vim -y
git clone https://github.com/spydr1/jackjack.git
pip install jackjack/
```

# Run Predictor

```
import glob
import tensorflow as tf

from jackjack.super_resolution.predictor as predictor
path = ''
image_file = tf.io.read_file(path)
test_img = tf.image.decode_png(image_file)
# you can check the list of pretrained-model. "predictor.DRCT.info()"
model = predictor.DRCT.get_pretrained_model(key="4xrealwebphoto_v4_drct-l")
z = model.predict(test_img)
```
