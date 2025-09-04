from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import modeling
import optimization
import tensorflow as tf
import tensorflow.compat.v1 as tf_v1
import glob
# from utils.utils import LogEvalRunHook
# from dllogger import Verbosity
# import math
# import numbers
# import numpy as np
# from tensorflow.core.protobuf import rewriter_config_pb2
# from utils.tb_utils import ExamplesPerSecondEstimatorHook, write_hparams_v1

curr_path = os.path.abspath(os.path.dirname(__file__))

flags = tf_v1.flags
FLAGS = flags.FLAGS

def init_flags():
  ## Required parameters
  flags.DEFINE_string(
      "bert_config_file", None,
      "The config json file corresponding to the pre-trained BERT model. "
      "This specifies the model architecture.")


  flags.DEFINE_integer("samples_between_eval", 150000, "MLPerf Evaluation frequency in samples.")

  flags.DEFINE_float("stop_threshold", 0.720, "MLperf Mask LM accuracy target")

  flags.DEFINE_integer("samples_start_eval", 3000000, " Required samples to start evaluation for MLPerf.")

  flags.DEFINE_bool("enable_device_warmup", False, " Enable device warmup for MLPerf.")

  flags.DEFINE_string(
      "input_files_dir", None,
      "Directory with input files, comma separated or single directory.")

  flags.DEFINE_string(
      "eval_files_dir", None,
      "Directory with eval files, comma separated or single directory. ")

  flags.DEFINE_string(
      "output_dir", None,
      "The output directory where the model checkpoints will be written.")

  ## Other parameters
  flags.DEFINE_string(
      "dllog_path", "/results/bert_dllog.json",
      "filename where dllogger writes to")

  flags.DEFINE_string(
      "init_checkpoint", None,
      "Initial checkpoint (usually from a pre-trained BERT model).")

  flags.DEFINE_string(
      "eval_checkpoint_path", None,
      "eval checkpoint path.")

  flags.DEFINE_bool(
      'is_dist_eval_enabled', False,  'IF true enable distributed evaluation')

  flags.DEFINE_string(
      "optimizer_type", "lamb",
      "Optimizer used for training - LAMB or ADAM")

  flags.DEFINE_integer(
      "max_seq_length", 512,
      "The maximum total input sequence length after WordPiece tokenization. "
      "Sequences longer than this will be truncated, and sequences shorter "
      "than this will be padded. Must match data generation.")

  flags.DEFINE_integer(
      "max_predictions_per_seq", 80,
      "Maximum number of masked LM predictions per sequence. "
      "Must match data generation.")

  flags.DEFINE_bool("do_train", False, "Whether to run training.")

  flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

  flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

  flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

  flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

  flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

  flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

  flags.DEFINE_integer("save_checkpoints_steps", 1000,
                      "How often to save the model checkpoint.")

  flags.DEFINE_integer("save_summary_steps", 1,
                       "How often to save the summary data.")

  flags.DEFINE_integer("display_loss_steps", 10,
                      "How often to print loss")

  flags.DEFINE_integer("iterations_per_loop", 1000,
                      "How many steps to make in each estimator call.")

  flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

  flags.DEFINE_integer("num_accumulation_steps", 1,
                      "Number of accumulation steps before gradient update."
                        "Global batch size = num_accumulation_steps * train_batch_size")

  flags.DEFINE_bool("allreduce_post_accumulation", False, "Whether to all reduce after accumulation of N steps or after each step")

  flags.DEFINE_bool(
      "verbose_logging", False,
      "If true, all of the trainable parameters are printed")

  flags.DEFINE_bool("horovod", False, "Whether to use Horovod for multi-gpu runs")

  flags.DEFINE_bool("report_loss", True, "Whether to report total loss during training.")

  flags.DEFINE_bool("manual_fp16", False, "Whether to use fp32 or fp16 arithmetic on GPU. "
                                          "Manual casting is done instead of using AMP")

  flags.DEFINE_bool("amp", True, "Whether to enable AMP ops. When false, uses TF32 on A100 and FP32 on V100 GPUS.")
  flags.DEFINE_bool("use_xla", True, "Whether to enable XLA JIT compilation.")
  flags.DEFINE_integer("init_loss_scale", 2**15, "Initial value of loss scale if mixed precision training")


def _decode_record(record, name_to_features):
    example = tf.io.parse_single_example(record, name_to_features)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t
    return example

# ========================================
# 1. Input Pipeline (unchanged)
# ========================================
def input_fn_builder(input_files,
                     batch_size,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
    name_to_features = {
        "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights": tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels": tf.io.FixedLenFeature([1], tf.int64),
    }

    if is_training:
        d = tf_v1.data.Dataset.from_tensor_slices(tf.constant(input_files))
        if FLAGS.horovod:
            d = d.shard(hvd.size(), hvd.rank())
        d = d.repeat()
        d = d.shuffle(buffer_size=len(input_files))
        cycle_length = min(num_cpu_threads, len(input_files))
        d = d.apply(
            tf_v1.data.experimental.parallel_interleave(
                tf_v1.data.TFRecordDataset,
                sloppy=is_training,
                cycle_length=cycle_length))
        d = d.shuffle(buffer_size=100)
    else:
        d = tf_v1.data.TFRecordDataset(input_files)
        d = d.repeat()

    d = d.apply(
        tf_v1.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True if is_training else False))
    return d


def gather_indexes(sequence_tensor, positions):
  """Gathers the vectors at the specific positions over a minibatch."""
  sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
  batch_size = sequence_shape[0]
  seq_length = sequence_shape[1]
  width = sequence_shape[2]

  flat_offsets = tf.reshape(
      tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
  flat_positions = tf.reshape(positions + flat_offsets, [-1])
  flat_sequence_tensor = tf.reshape(sequence_tensor,
                                    [batch_size * seq_length, width])
  output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
  return output_tensor

def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  input_tensor = gather_indexes(input_tensor, positions)

  with tf_v1.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    with tf_v1.variable_scope("transform"):
      input_tensor_dense_layer = tf.keras.layers.Dense(
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(bert_config.initializer_range)
            )
      input_tensor = input_tensor_dense_layer(input_tensor)

      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf_v1.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf_v1.zeros_initializer())
    logits = tf.matmul(tf.cast(input_tensor, tf.float32), output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)

    log_probs = tf.nn.log_softmax(logits - tf.reduce_max(input_tensor=logits, keepdims=True, axis=-1), axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    per_example_loss = -tf.reduce_sum(input_tensor=log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(input_tensor=label_weights * per_example_loss)
    denominator = tf.reduce_sum(input_tensor=label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  with tf_v1.variable_scope("cls/seq_relationship"):
    output_weights = tf_v1.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf_v1.get_variable(
        "output_bias", shape=[2], initializer=tf_v1.zeros_initializer())

    logits = tf.matmul(tf.cast(input_tensor, tf.float32), output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits - tf.reduce_max(input_tensor=logits, keepdims=True, axis=-1), axis=-1)


    labels = tf.reshape(labels, [-1])
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    per_example_loss = -tf.reduce_sum(input_tensor=one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(input_tensor=per_example_loss)
    return (loss, per_example_loss, log_probs)

# ========================================
# 3. Main Training Script (No Estimator)
# ========================================
def main(_):
    tf_v1.logging.set_verbosity(tf_v1.logging.INFO)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tf.io.gfile.makedirs(FLAGS.output_dir)

    input_files = []
    for input_file_dir in FLAGS.input_files_dir.split(","):
        input_files.extend(glob.glob(os.path.join(input_file_dir, "*")))

    if FLAGS.horovod and len(input_files) < hvd.size():
        input_files = [input_files[0]] * hvd.size()

    # ========================================
    # 4. Build Graph
    # ========================================
    tf_v1.reset_default_graph()

    # Placeholders for features (not needed if using dataset iterator directly)
    train_dataset = input_fn_builder(
        input_files=input_files,
        batch_size=FLAGS.train_batch_size,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True
    )
    iterator = tf_v1.data.make_one_shot_iterator(train_dataset)
    features = iterator.get_next()

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]

    # Build model
    model = modeling.BertModel(
        config=bert_config,
        is_training=True,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False,
        compute_type=tf.float16 if FLAGS.manual_fp16 else tf.float32
    )

    # Loss
    (masked_lm_loss, _, _) = get_masked_lm_output(
        bert_config, model.get_sequence_output(), model.get_embedding_table(),
        masked_lm_positions, masked_lm_ids, masked_lm_weights)
    (next_sentence_loss, _, _) = get_next_sentence_output(
        bert_config, model.get_pooled_output(), next_sentence_labels)

    total_loss = tf.identity(masked_lm_loss + next_sentence_loss, name='total_loss')

    # Optimizer
    if FLAGS.optimizer_type == "lamb":
        weight_decay_rate = 0.01
        beta_1, beta_2, epsilon, power = 0.9, 0.999, 1e-6, 1
    else:
        weight_decay_rate = beta_1 = beta_2 = epsilon = power = None

    learning_rate = FLAGS.learning_rate * (hvd.size() if FLAGS.horovod else 1)

    # Create optimizer (returns train_op and loss_scale_var if AMP)
    train_op = optimization.create_optimizer(
        loss=total_loss,
        init_lr=learning_rate,
        num_train_steps=FLAGS.num_train_steps,
        num_warmup_steps=0,
        manual_fp16=FLAGS.manual_fp16,
        use_fp16=FLAGS.amp,
        num_accumulation_steps=FLAGS.num_accumulation_steps,
        optimizer_type=FLAGS.optimizer_type,
        allreduce_post_accumulation=FLAGS.allreduce_post_accumulation,
        init_loss_scale=FLAGS.init_loss_scale,
        weight_decay_rate=weight_decay_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        power=power,
        hvd=hvd if FLAGS.horovod else None
    )
    # Horovod: broadcast initial variables
    bcast_op = hvd.broadcast_global_variables(0) if FLAGS.horovod else None


    # Checkpoint and summary
    global_step = tf_v1.train.get_or_create_global_step()
    saver = tf_v1.train.Saver(max_to_keep=5)

    # Session config
    session_config = tf_v1.ConfigProto()
    session_config.allow_soft_placement = True
    session_config.log_device_placement = False
    session_config.gpu_options.allow_growth = True
    print(f"size:{hvd.size()}, local_rank:{hvd.local_rank()}")

    # Training loop
    with tf_v1.Session(config=session_config) as sess:
        sess.run(tf_v1.global_variables_initializer())

        # Horovod: broadcast initial weights
        if FLAGS.horovod:
            sess.run(bcast_op)

        tf_v1.logging.info("***** Running training *****")
        tf_v1.logging.info("  Batch size = %d", FLAGS.train_batch_size)

        step = 0
        total_time = 0
        total_setences = 0
        acc_steps = 0
        while step < FLAGS.num_train_steps:
            try:
                start_time = time.time()
                _, step, total_loss_val, nsp_loss_val, mlm_loss_val = sess.run([train_op, global_step, total_loss, next_sentence_loss, masked_lm_loss])

                total_time += time.time() - start_time
                total_setences += FLAGS.train_batch_size*(hvd.size() if FLAGS.horovod else 1)

                if step % FLAGS.display_loss_steps == 0 and acc_steps % FLAGS.num_accumulation_steps == 0:
                    print(f"Step {step}, total_loss: {total_loss_val:.4f}, nsp_loss: {nsp_loss_val:.4f}, mlm_loss: {mlm_loss_val:.4f}, sentences_per_second: {total_setences/total_time:.4f}")
                
                acc_steps += 1
            except tf.errors.OutOfRangeError:
                break

        if hvd:
            hvd.join()

if __name__ == "__main__":
    start_time = time.time()
    init_flags()
    try:
        from dltest import show_training_arguments
        show_training_arguments(FLAGS)
    except:
        pass
    print("*****************************************")
    print("Arguments passed to this program: run_pretraining.")
    tf_v1.disable_v2_behavior()
    tf_v1.disable_eager_execution()
    tf_v1.enable_resource_variables()
    if FLAGS.horovod:
        import horovod.tensorflow as hvd
        hvd.init()
        os.environ['CUDA_VISIBLE_DEVICES'] = str(hvd.local_rank())
    #load_habana_module()
    flags.mark_flag_as_required("input_files_dir")
    if FLAGS.do_eval:
        flags.mark_flag_as_required("eval_files_dir")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    if FLAGS.use_xla and FLAGS.manual_fp16:
        print('WARNING! Combining --use_xla with --manual_fp16 may prevent convergence.')
        print('         This warning message will be removed when the underlying')
        print('         issues have been fixed and you are running a TF version')
        print('         that has that fix.')
    tf_v1.app.run()
    end_time = time.time()
    e2e_time = end_time - start_time
    print("e2e_time:",e2e_time)
