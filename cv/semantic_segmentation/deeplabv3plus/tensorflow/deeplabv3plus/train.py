# Copyright(c)2023,ShanghaiIluvatarCoreXSemiconductorCo.,Ltd.
# AllRightsReserved.
#
#   LicensedundertheApacheLicense,Version2.0(the"License");youmay
#   otusethisfileexceptincompliancewiththeLicense.Youmayobtain
#   acopyoftheLicenseat
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#   Unlessrequiredbyapplicablelaworagreedtoinwriting,software
#   distributedundertheLicenseisdistributedonan"ASIS"BASIS,WITHOUT
#   WARRANTIESORCONDITIONSOFANYKIND,eitherexpressorimplied.Seethe
#   Licenseforthespecificlanguagegoverningpermissionsandlimitations
#   undertheLicense.

"""Module providing Trainer class for deeplabv3plus"""

import os
import time
import tensorflow as tf

import wandb
from wandb.keras import WandbCallback

from deeplabv3plus.datasets import GenericDataLoader
from deeplabv3plus.model import DeeplabV3Plus

class TimeCount(tf.keras.callbacks.Callback):
    def __init__(self,img_nums):
        super().__init__()
        self.time_acc = 0
        self.start_time = 0 
        self.img_nums = img_nums

    def on_epoch_end(self, epoch, logs=None):
        print('avg fps: {}'.format(self.img_nums / self.time_acc))
        print("time cost {}".format(self.time_acc))
        self.time_acc = 0

    def on_batch_begin(self, epoch, logs=None):
        self.start_time = time.time()

    def on_batch_end(self, epoch, logs=None):
        self.time_acc += (time.time() - self.start_time)
        self.start_time = time.time()

class Trainer:
    """Class for managing DeeplabV3+ model training.

    Args:
        config:
            python dictionary containing training configuration for
            DeeplabV3Plus
    """
    def __init__(self, config):
        self.config = config
        self._assert_config()

        # Train Dataset
        train_dataloader = GenericDataLoader(self.config[
            'train_dataset_config'])
        self.train_data_length = len(train_dataloader)
        print('[+] Data points in train dataset: {}'.format(
            self.train_data_length))
        self.train_dataset = train_dataloader.get_dataset()
        print('Train Dataset:', self.train_dataset)

        # Validation Dataset
        val_dataloader = GenericDataLoader(self.config[
            'val_dataset_config'])
        self.val_data_length = len(val_dataloader)
        print('Data points in train dataset: {}'.format(
            self.val_data_length))
        self.val_dataset = val_dataloader.get_dataset()
        print('Val Dataset:', self.val_dataset)

        self._model = None
        self._wandb_initialized = False

    @property
    def model(self):
        """Property returning model being trained."""

        if self._model is not None:
            return self._model

        self._model = DeeplabV3Plus(
            num_classes=self.config['num_classes'],
            backbone=self.config['backbone']
        )

        SparseCategoricalCrossentropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        def my_loss(y_true, y_pred):
            b,h,w,c = y_pred.shape
            mask = tf.less(y_true,self.config['num_classes']+1)
            y_true_mask = tf.boolean_mask(y_true,mask)
            y_pred_mask = tf.boolean_mask(y_pred,tf.reshape(mask,[b,h,w]))
            return SparseCategoricalCrossentropy(y_true_mask,y_pred_mask)

        self._model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.config['learning_rate']
            ),
            loss = [my_loss],
            metrics=['accuracy']
        )

        return self._model

    @staticmethod
    def _assert_dataset_config(dataset_config):
        assert 'images' in dataset_config and \
            isinstance(dataset_config['images'], list)
        assert 'labels' in dataset_config and \
            isinstance(dataset_config['labels'], list)

        assert 'height' in dataset_config and \
            isinstance(dataset_config['height'], int)
        assert 'width' in dataset_config and \
            isinstance(dataset_config['width'], int)

        assert 'batch_size' in dataset_config and \
            isinstance(dataset_config['batch_size'], int)

    def _assert_config(self):
        assert 'project_name' in self.config and \
            isinstance(self.config['project_name'], str)
        assert 'experiment_name' in self.config and \
            isinstance(self.config['experiment_name'], str)

        assert 'train_dataset_config' in self.config
        Trainer._assert_dataset_config(self.config['train_dataset_config'])
        assert 'val_dataset_config' in self.config
        Trainer._assert_dataset_config(self.config['val_dataset_config'])

        assert 'num_classes' in self.config and \
            isinstance(self.config['num_classes'], int)
        assert 'backbone' in self.config and \
            isinstance(self.config['backbone'], str)

        assert 'learning_rate' in self.config and \
            isinstance(self.config['learning_rate'], float)

        assert 'checkpoint_dir' in self.config and \
            isinstance(self.config['checkpoint_dir'], str)
        assert 'checkpoint_file_prefix' in self.config and \
            isinstance(self.config['checkpoint_file_prefix'], str)

        assert 'epochs' in self.config and \
            isinstance(self.config['epochs'], int)

    def connect_wandb(self):
        """Connects Trainer to wandb.

        Runs wandb.init() with the given wandb_api_key, project_name and
        experiment_name.
        """
        if 'wandb_api_key' not in self.config:
            return

        os.environ['WANDB_API_KEY'] = self.config['wandb_api_key']
        wandb.init(
            project=self.config['project_name'],
            name=self.config['experiment_name']
        )
        self._wandb_initialized = True

    def _get_checkpoint_filename_format(self):
        if self.config['checkpoint_dir'] == 'wandb://':
            if 'wandb_api_key' not in self.config:
                raise ValueError("Invalid configuration, wandb_api_key not "
                                 "provided!")
            if not self._wandb_initialized:
                raise ValueError("Wandb not intialized, "
                                 "checkpoint_filename_format is unusable.")

            return os.path.join(wandb.run.dir,
                                self.config['checkpoint_file_prefix'] +
                                "{epoch}")

        return os.path.join(self.config['checkpoint_dir'],
                            self.config['checkpoint_file_prefix'] +
                            "{epoch}")

    def _get_logger_callback(self):
        if 'wandb_api_key' not in self.config:
            return tf.keras.callbacks.TensorBoard()

        try:
            return WandbCallback(save_weights_only=True, save_model=False)
        except wandb.Error as error:
            if 'wandb_api_key' in self.config:
                raise error  # rethrow

            print("[-] Defaulting to TensorBoard logging...")
            return tf.keras.callbacks.TensorBoard()

    def train(self):
        """Trainer entry point.

        Attempts to connect to wandb before starting training. Runs .fit() on
        loaded model.
        """
        if not self._wandb_initialized:
            self.connect_wandb()

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=self._get_checkpoint_filename_format(),
                monitor='val_loss',
                save_best_only=True,
                mode='min',
                save_weights_only=True
            ),

            self._get_logger_callback(),
            TimeCount(self.train_data_length)
        ]

        history = self.model.fit(
            self.train_dataset, validation_data=self.val_dataset,

            steps_per_epoch=self.train_data_length //
            self.config['train_dataset_config']['batch_size'],

            validation_steps=self.val_data_length //
            self.config['val_dataset_config']['batch_size'],

            epochs=self.config['epochs'], callbacks=callbacks
        )

        return history
