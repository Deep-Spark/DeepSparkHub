"""split criteo data in mlperf format"""
import math
import os
import numpy as np
from tqdm import tqdm
from absl import app
from absl import logging
from absl import flags


FLAGS = flags.FLAGS

flags.DEFINE_enum("stage", "val", ["train", "val", "test"], "")
flags.DEFINE_string("bin_data_root", "", "path to binary criteo dataset")
flags.DEFINE_string("out_root", "", "path where to save")


def main(argv):
    bin_data = os.path.join(FLAGS.bin_data_root, F"{FLAGS.stage}_data.bin")
    out_path = os.path.join(FLAGS.out_root, FLAGS.stage)
    logging.info("processing %s", bin_data)

    if not os.path.exists(out_path):
        logging.info("Creating %s", out_path)
        os.mkdir(out_path)

    input_data_f = open(bin_data, "rb")
    numerical_f = open(os.path.join(out_path, "numerical.bin"), "wb+")
    label_f = open(os.path.join(out_path, 'label.bin'), 'wb+')
    categorical_fs = []
    for i in range(26):
        categorical_fs.append(open(os.path.join(out_path, F'cat_{i}.bin'), 'wb+'))

    bytes_per_entry = 160
    total_size = os.path.getsize(bin_data)
    block = 16384
    for i in tqdm(range(math.ceil((total_size // bytes_per_entry) / block))):
        raw_data = np.frombuffer(input_data_f.read(bytes_per_entry * block), dtype=np.int32)
        if raw_data.shape[0] != 655360:
            print(raw_data.shape)
        batch_data = raw_data.reshape(-1, 40)

        numerical_features = batch_data[:, 1:14]
        numerical_f.write(numerical_features.astype(np.float32).tobytes())

        label = batch_data[:, 0]
        label_f.write(label.astype(np.float32).tobytes())
        for i in range(26):
            categorical_fs[i].write(batch_data[:, (i + 14):(i + 15)].tobytes())

    input_data_f.close()
    numerical_f.close()
    label_f.close()
    for f in categorical_fs:
        f.close()

if __name__ == '__main__':
    app.run(main)
