"""Test dataset"""
import os
import time

import numpy as np

from absl import flags
from absl import logging
from absl.testing import absltest

import torch

from dlrm.data import dataset

FLAGS = flags.FLAGS

flags.DEFINE_string("bin_dataset", None, "Full path to reference binary dataset")
flags.DEFINE_string("dist_dataset", None, "Full path to split binary dataset")
flags.DEFINE_integer("batch_size", 16384, "")
flags.DEFINE_integer("num_batches", 500, "Number of batches to test.")
flags.DEFINE_boolean("shuffle_batch_order", False, "Read batch in train dataset by random order", short_name="shuffle")

# pylint:disable=missing-docstring, no-self-use

class RefCriteoTerabyteLoaderTest(absltest.TestCase):

    def test_dataloader(self):
        """Test reference binary data loader

        It tests data loader function also benchmark performance. It does NOT verify correctness of the dataset
        """
        batch_size = FLAGS.batch_size
        num_batches = FLAGS.num_batches
        dataset_test = dataset.CriteoBinDataset(
            data_file=FLAGS.bin_dataset,
            batch_size=batch_size,
            shuffle=FLAGS.shuffle)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=None, num_workers=0, pin_memory=False, collate_fn=dataset.data_collate_fn)

        if num_batches > len(data_loader_test):
            logging.warning(F"Only {len(data_loader_test)} batches in the dataset, asked for {num_batches}")
            num_batches = len(data_loader_test)

        for i, _ in enumerate(data_loader_test):
            if i == 0:
                start_time = time.time()
            if i % 100 == 0:
                partial_time = time.time()
                logging.info("Fetched %d batches in %.2fs", i, partial_time - start_time)
            if i > num_batches:
                break
        end_time = time.time()
        logging.info("Finished fetching %d records at %.1f records/s",
                     num_batches * batch_size,
                     num_batches * batch_size / (end_time - start_time))

class CriteoMemmapLoaderTest(absltest.TestCase):

    def test_dataloader(self):
        """Test reference binary data loader

        It tests data loader function also benchmark performance. It does NOT verify correctness of the dataset
        """
        batch_size = FLAGS.batch_size
        num_batches = FLAGS.num_batches
        dataset_test = dataset.CriteoMemmapDataset(
            data_file=FLAGS.bin_dataset,
            batch_size=batch_size,
            shuffle=FLAGS.shuffle)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=None, num_workers=0, pin_memory=False, collate_fn=dataset.data_collate_fn)

        if num_batches > len(data_loader_test):
            logging.warning(F"Only {len(data_loader_test)} batches in the dataset, asked for {num_batches}")
            num_batches = len(data_loader_test)

        for i, _ in enumerate(data_loader_test):
            if i == 0:
                start_time = time.time()
            if i % 100 == 0:
                partial_time = time.time()
                logging.info("Fetched %d batches in %.2fs", i, partial_time - start_time)
            if i > num_batches:
                break
        end_time = time.time()
        logging.info("Finished fetching %d records at %.1f records/s",
                     num_batches * batch_size,
                     num_batches * batch_size / (end_time - start_time))


class DistCriteoDatasetTest(absltest.TestCase):

    def test_creation(self):
        _ = dataset.DistCriteoDataset(
            data_path=FLAGS.dist_dataset,
            batch_size=FLAGS.batch_size,
            numerical_features=True,
            categorical_features=[0, 1, 2, 3, 4])

    def test_against_bin(self):
        dist_dataset = dataset.DistCriteoDataset(
            FLAGS.dist_dataset,
            batch_size=FLAGS.batch_size,
            numerical_features=True,
            categorical_features=range(26))
        bin_dataset = dataset.CriteoBinDataset(
            FLAGS.bin_dataset,
            batch_size=FLAGS.batch_size)

        data_loader_dist = torch.utils.data.DataLoader(
            dist_dataset, batch_size=None, num_workers=0, pin_memory=False, collate_fn=dataset.data_collate_fn)

        data_loader_bin = torch.utils.data.DataLoader(
            bin_dataset, batch_size=None, num_workers=0, pin_memory=False, collate_fn=dataset.data_collate_fn)

        for i, (data, ref) in enumerate(zip(data_loader_dist, data_loader_bin)):
            if i > FLAGS.num_batches:
                break
            np.testing.assert_equal(data[0].cpu().numpy(), ref[0].cpu().numpy(), err_msg=F"Miss match in batch {i}.")
            np.testing.assert_equal(data[1].cpu().numpy(), ref[1].cpu().numpy(), err_msg=F"Miss match in batch {i}.")
            np.testing.assert_equal(data[2].cpu().numpy(), ref[2].cpu().numpy(), err_msg=F"Miss match in batch {i}.")

    def test_dataloader(self):
        batch_size = FLAGS.batch_size
        num_batches = FLAGS.num_batches

        try:
            rank = int(os.environ["RANK"])
        except KeyError:
            rank = 0

        if rank == 0:
            numerical_features = True
            categorical_features = None
        else:
            numerical_features = False
            categorical_features = range(rank * 4, (rank + 1) * 4)

        dataset_test = dataset.DistCriteoDataset(
            data_path=FLAGS.dist_dataset,
            batch_size=batch_size,
            shuffle=FLAGS.shuffle,
            numerical_features=numerical_features,
            categorical_features=categorical_features)

        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=None, num_workers=0, pin_memory=False, collate_fn=dataset.data_collate_fn)

        if num_batches > len(data_loader_test):
            logging.warning(F"Only {len(data_loader_test)} batches in the dataset, asked for {num_batches}")
            num_batches = len(data_loader_test)

        for i, data_batch in enumerate(data_loader_test):
            if i == 0:
                start_time = time.time()
            if i % 100 == 0:
                partial_time = time.time()
                logging.info("Fetched %d batches in %.2fs", i, partial_time - start_time)
            if i > num_batches:
                break
        end_time = time.time()
        logging.info("Finished fetching %d records at %.1f records/s",
                     num_batches * batch_size,
                     num_batches * batch_size / (end_time - start_time))


if __name__ == '__main__':
    absltest.main()
