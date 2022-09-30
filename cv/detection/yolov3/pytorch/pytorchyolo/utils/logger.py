import os
import datetime
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    class SummaryWriter(object):
        def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10,
                     flush_secs=120, filename_suffix=''):
            if not log_dir:
                import socket
                from datetime import datetime
                current_time = datetime.now().strftime('%b%d_%H-%M-%S')
                log_dir = os.path.join(
                    'runs', current_time + '_' + socket.gethostname() + comment)
            self.log_dir = log_dir
            self.purge_step = purge_step
            self.max_queue = max_queue
            self.flush_secs = flush_secs
            self.filename_suffix = filename_suffix

            # Initialize the file writers, but they can be cleared out on close
            # and recreated later as needed.
            self.file_writer = self.all_writers = None
            self._get_file_writer()

            # Create default bins for histograms, see generate_testdata.py in tensorflow/tensorboard
            v = 1E-12
            buckets = []
            neg_buckets = []
            while v < 1E20:
                buckets.append(v)
                neg_buckets.append(-v)
                v *= 1.1
            self.default_bins = neg_buckets[::-1] + [0] + buckets

        def _check_caffe2_blob(self, item): pass

        def _get_file_writer(self): pass

        def get_logdir(self):
            """Returns the directory where event files will be written."""
            return self.log_dir

        def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None, run_name=None): pass

        def add_scalar(self, tag, scalar_value, global_step=None, walltime=None, new_style=False): pass

        def add_scalars(self, main_tag, tag_scalar_dict, global_step=None, walltime=None): pass

        def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None): pass

        def add_histogram_raw(self, tag, min, max, num, sum, sum_squares, bucket_limits, bucket_counts, global_step=None, walltime=None): pass

        def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'): pass

        def add_images(self, tag, img_tensor, global_step=None, walltime=None, dataformats='NCHW'): pass

        def add_image_with_boxes(self, tag, img_tensor, box_tensor, global_step=None, walltime=None, rescale=1, dataformats='CHW', labels=None): pass

        def add_figure(self, tag, figure, global_step=None, close=True, walltime=None): pass

        def add_video(self, tag, vid_tensor, global_step=None, fps=4, walltime=None): pass

        def add_audio(self, tag, snd_tensor, global_step=None, sample_rate=44100, walltime=None): pass

        def add_text(self, tag, text_string, global_step=None, walltime=None): pass

        def add_onnx_graph(self, prototxt): pass

        def add_graph(self, model, input_to_model=None, verbose=False): pass

        @staticmethod
        def _encode(rawstr): pass

        def add_embedding(self, mat, metadata=None, label_img=None, global_step=None, tag='default', metadata_header=None): pass

        def add_pr_curve(self, tag, labels, predictions, global_step=None, num_thresholds=127, weights=None, walltime=None): pass

        def add_pr_curve_raw(self, tag, true_positive_counts, false_positive_counts, true_negative_counts, false_negative_counts, precision, recall, global_step=None, num_thresholds=127, weights=None, walltime=None): pass

        def add_custom_scalars_multilinechart(self, tags, category='default', title='untitled'): pass

        def add_custom_scalars_marginchart(self, tags, category='default', title='untitled'): pass

        def add_custom_scalars(self, layout): pass

        def add_mesh(self, tag, vertices, colors=None, faces=None, config_dict=None, global_step=None, walltime=None): pass

        def flush(self): pass

        def close(self): pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            self.close()



class Logger(object):
    def __init__(self, log_dir, log_hist=True):
        """Create a summary writer logging to log_dir."""
        if log_hist:    # Check a new folder for each log should be dreated
            log_dir = os.path.join(
                log_dir,
                datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def list_of_scalars_summary(self, tag_value_pairs, step):
        """Log scalar variables."""
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, step)
