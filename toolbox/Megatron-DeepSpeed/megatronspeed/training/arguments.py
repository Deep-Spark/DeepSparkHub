import os
from functools import wraps
import argparse
import warnings

import torch

from megatron.training.arguments import load_retro_args, _print_args, _check_arg_is_not_none
import deepspeed
from packaging import version
from megatron.core.utils import is_torch_min_version, get_torch_version
from megatron.core.transformer.enums import AttnBackend
from megatron.training.utils import update_use_dist_ckpt

def extra_args_provider_decorator(extra_args_provider):
    @wraps(extra_args_provider)
    def wrapper(parser):
        if extra_args_provider is not None:
            parser = extra_args_provider(parser)
        parser = process_args(parser)
        return parser

    return wrapper


def parse_args_wrapper(parse_args):
    @wraps(parse_args)
    def wrapper(extra_args_provider=None, ignore_unknown_args=False):
        decorated_provider = extra_args_provider_decorator(extra_args_provider)
        args = parse_args(decorated_provider, ignore_unknown_args)

        # helper argument to set deepspeed pipeline parallel or not
        args.ds_pipeline_enabled = not args.no_pipeline_parallel

        return args

    return wrapper


def process_args(parser):
    parser.conflict_handler = 'resolve'
    parser = _add_network_size_args(parser)
    parser = _add_regularization_args(parser)
    parser = _add_training_args(parser)
    parser = _add_learning_rate_args(parser)
    parser = _add_checkpointing_args(parser)
    parser = _add_mixed_precision_args(parser)
    parser = _add_distributed_args(parser)
    parser = _add_data_args(parser)
    parser = _add_logging_args(parser)
    parser = _add_zero_args(parser)
    parser = _add_memoryopt_args(parser)
    parser = _add_activation_checkpoint_args(parser)
    parser = _add_transformer_engine_args(parser)
    parser = _add_retro_args(parser)

    parser = deepspeed.add_config_arguments(parser)

    return parser


def _add_transformer_engine_args(parser):
    group = parser.add_argument_group(title='Transformer-Engine')

    # group.add_argument('--fp8-e4m3', action='store_true',
    #                     help='E4M3 TransformerLayer', dest='fp8_e4m3')
    # group.add_argument('--fp8-hybrid', action='store_true',
    #                     help='Hybrid FP8 TransformerLayer', dest='fp8_hybrid')
    group.add_argument('--transformer-impl', default='local',
                       choices=['local', 'transformer_engine'],
                       help='Which Transformer implementation to use.')
    return parser

def _add_network_size_args(parser):
    group = parser.add_argument_group(title='network size')

    group.add_argument('--ds-num-experts', type=int, nargs='+', default=[1,],
                           help='number of experts list, MoE related.')
    group.add_argument('--mlp-type', type=str, default='standard',
                           help='Only applicable when num-experts > 1, accepts [standard, residual]')
    group.add_argument('--topk', type=int, default=1,
                           help='Sets the k in TopK gating for MoE layers')
    group.add_argument('--expert-interval', type=int, default=1,
                           help='Use experts in every "expert-interval" layers')
    group.add_argument('--num-key-value-heads', type=int, default=None,
                       help='Number of key_value heads that should be used to implement Grouped Query Attention.')
    group.add_argument('--rotary-position-embeddings-theta', type=int, default=10000,
                       help='Rotary positional embeddings theta value.',
                       dest='rope_theta')
    group.add_argument('--layernorm-epsilon', type=float, default=1e-5,
                       help='Layer norm epsilon.')
    group.add_argument('--disable-mem-efficient-ln', action='store_false', 
                       help='Disable the memory-efficient fused LayerNorm optimization '
                       'introduced in https://github.com/NVIDIA/apex/pull/1715', dest='mem_efficient_ln')
    group.add_argument('--num-experts-switch', type=int, default=None,
                       help='Number of Experts in Switch Transformer (None means no Switch)')
    group.add_argument('--embedding-weights-in-fp32', action='store_true',
                       help='Cast word embedding weights to fp32 before embedding fwd.'),
    group.add_argument('--kill-switch-file', type=str, default=None,
                       help='Location of kill switch file. ' 
                            'If found will automatically exit the program at runtime.')
    return parser

def _add_logging_args(parser):
    group = parser.add_argument_group(title='logging')

    group.add_argument('--log-optimizer-states-to-tensorboard',
                       action='store_true',
                       help='If set, write various optimizer states to '
                       'tensorboard. This feature may consume extra GPU memory.')
    return parser

def _add_regularization_args(parser):
    group = parser.add_argument_group(title='regularization')

    group.add_argument('--actor-weight-decay', type=float, default=0.01,
                       help='RLHF actor model weight decay coefficient for L2 regularization.')
    group.add_argument('--critic-weight-decay', type=float, default=0.01,
                       help='RLHF critic model weight decay coefficient for L2 regularization.')
    return parser

def _add_training_args(parser):
    group = parser.add_argument_group(title='training')

    group.add_argument('--rlhf-train-mbs', type=int, default=None,
                       help='Micro batch size in RLHF train time')
    group.add_argument('--custom-recompute-layers-per-stage', nargs='*', type=int, default=None,
                       help='custom recompute num layers in each PP stage, it should be equal to PP size ')
    group.add_argument('--enable-zbh1-pipeline', action='store_true',
                       help='Activate zero bubble pipeline parallelism schedule method')
    group.add_argument('--enable-zbh1-exact-semantics', action='store_true',
                       help='Use an exact semantics for zbh1 schedule, might be slower than the default.')


    # deprecated
    # HACK: added back arguments because DeepSpeed still relies on the old
    # activation checkpointing mechanism.
    group.add_argument('--distribute-checkpointed-activations',
                       action='store_true',
                       help='If set, distribute checkpointed activations '
                       'across model parallel group.')
    group.add_argument('--checkpoint-num-layers', type=int, default=1,
                       help='chunk size (number of layers) for checkpointing.')
    group.add_argument('--train-tokens', type=int, default=None,
                       help='Total number of tokens to train over all '
                       'training runs.')
    group.add_argument('--random-ltd',
                       action='store_true',
                       help='enable random layer token drop')
    group.add_argument('--disable-moe-token-dropping', action='store_false',
                       help='Disable MoE expert token dropping.',
                       dest='moe_token_dropping')
    group.add_argument('--moe-train-capacity-factor', type=float, default=1.0,
                       help='The capacity of the MoE expert at training time')
    group.add_argument('--moe-eval-capacity-factor', type=float, default=1.0,
                       help='The capacity of the MoE expert at eval time.')
    group.add_argument('--moe-min-capacity', type=int, default=4,
                       help='The minimum capacity per MoE expert regardless of the capacity_factor.')
    group.add_argument('--moe-loss-coeff', type=float, default=0.1,
                       help='Scaling coefficient for adding MoE loss to model loss')
    group.add_argument('--create-moe-param-group', action='store_true',
                       help='Create separate groups for MoE params.'
                       'This is necessary for techniques like ZeRO.')
    group.add_argument('--disable-moe-top2-2nd-expert-sampling', action='store_false',
                       help='Disable MoE top2 sampling of the 2nd expert. Instead of sampling, use argmax.',
                       dest='moe_top2_2nd_expert_sampling')
    group.add_argument('--use-flash-attn-v1', dest='use_flash_attn_v1', action='store_true',
                       help='use first version FlashAttention implementation of attention. '
                       'https://arxiv.org/abs/2205.14135')
    group.add_argument('--use-flash-attn-v2', action='store_true',
                       help='use second version FlashAttention implementation of attention. '
                       'https://arxiv.org/abs/2307.08691')
    group.add_argument('--use-flash-attn-triton', action='store_true',
                       help='use FlashAttention implementation of attention using Triton.')
    group.add_argument('--use-flash-attn-builder', action='store_true',
                       help='use FlashAttention op builder.')
    group.add_argument('--ds-inference', action='store_true',
                       help='DeepSpeed inference engine being used')
    group.add_argument('--cpu-optimizer', action='store_true',
                       help='Run optimizer on CPU')
    group.add_argument('--cpu_torch_adam', action='store_true',
                       help='Use Torch Adam as optimizer on CPU.')
    group.add_argument('--ds_fused_adam', action='store_true',
                       help='Use DeepSpeed FusedAdam as optimizer.')
    group.add_argument('--no-pipeline-parallel', action='store_true',
                       help='Disable Deepspeed pipeline parallelism')
    group.add_argument('--use-tutel', action='store_true',
                       help='Use Tutel optimization for MoE')
    group.add_argument('--inference', action='store_true',
                       help='Very basic inference mode: not allocating optim/lr - requires ZERO_STAGE=0')
    group.add_argument('--ds-sequence-parallel-size', type=int, default=1,
                       help='Enable DeepSpeed\'s sequence parallel. Cannot be combined with "--sequence-parallel", which enables Megatron-LM\'s sequence parallel.')
    group.add_argument('--force-ds-sequence-parallel', action='store_true',
                       help='use DeepSpeed sequence parallelism regardless of sequence parallel size.')
    group.add_argument('--use-dataset-only', type=bool, required=False, default=False,
                       help='If set to True, only use the megatron dataset for external trainer ')
    group.add_argument('--RLHF', action="store_true",
                       help='RLHF mode')
    group.add_argument('--ppo-epoches', type=int, default=1,
                       help='RLHF model train epoches')
    return parser

def _add_learning_rate_args(parser):
    group = parser.add_argument_group(title='learning rate')

    group.add_argument('--actor-learning-rate', type=float, default=None,
                       help='Initial RLHF actor model learning rate. Depending on decay style '
                       'and initial warmup, the learing rate at each '
                       'iteration would be different.')
    group.add_argument('--critic-learning-rate', type=float, default=None,
                       help='Initial RLHF critic model learning rate. Depending on decay style '
                       'and initial warmup, the learing rate at each '
                       'iteration would be different.')
    group.add_argument('--lr-decay-tokens', type=int, default=None,
                       help='number of tokens to decay learning rate over,'
                       ' If not None will override iter/sample-based decay')
    group.add_argument('--lr-warmup-tokens', type=int, default=None,
                       help='number of tokens to linearly warmup '
                       'learning rate over.')
    return parser

def _add_checkpointing_args(parser):
    group = parser.add_argument_group(title='checkpointing')

    group.add_argument('--load-tag', type=str, default=None,
                       help='Specific checkpoint tag to load. Ignores latest.')
    parser.add_argument("--actor_model_name_or_path", type=str, default=None,
                        help="Directory containing a actor_model checkpoint.")
    parser.add_argument("--critic_model_name_or_path", type=str, default=None,
                        help="Directory containing a critic_model checkpoint.")
    group.add_argument('--no-load-lr-state', action='store_true',
                       help='Do not load lr state when loading checkpoint.')
    group.add_argument('--universal-checkpoint', action='store_true',
                        help='Loading a universal format checkpoint.')
    return parser

def _add_mixed_precision_args(parser):
    group = parser.add_argument_group(title='mixed precision')

    group.add_argument('--no-query-key-layer-scaling', action='store_false',
                       help='Do not scale Q * K^T by 1 / layer-number.',
                       dest='apply_query_key_layer_scaling')
    return parser

def _add_distributed_args(parser):
    group = parser.add_argument_group(title='distributed')

    group.add_argument('--enable-expert-tensor-parallelism', action='store_true',
                        default=False,
                        help="use tensor parallelism for expert layers in MoE")
    group.add_argument('--partition-method',
                       type=str, default='type:transformer',
                       help='use deepspeed to patition layers. method include: uniform, parameters, type:transformer, custom')
    group.add_argument('--custom-partition', nargs='*',
                       type=int, default=None,
                       help='customized model layers to PP stages, parameter of partition-method should set < custom > to take this effect. \
                       example: divide 32 layers to 6 PP stages: 5 5 5 6 6 5. it means there are 5/5/5/6/6/5 layers in 6 pp stages')
    group.add_argument('--moe-expert-parallel-size', type=int, default=1,
                       help='Degree of the MoE expert parallelism.')
    group.add_argument('--DDP-impl', default='local',
                       choices=['local', 'torch', 'FSDP'],
                       help='which DistributedDataParallel implementation '
                       'to use.')
    group.add_argument('--no-contiguous-buffers-in-local-ddp',
                       action='store_false', help='If set, dont use '
                       'contiguous buffer in local DDP.',
                       dest='use_contiguous_buffers_in_local_ddp')
    group.add_argument('--pp-delay', action='store_true', 
                       default=False, help='')
    group.add_argument('--pp-split-size', type=int, default=1,
                    help='')
    return parser
    
def _add_data_args(parser):
    group = parser.add_argument_group(title='data and dataloader')

    group.add_argument('--aml-data-download-path', type=str, default=None,
                       help='Path to mounted input dataset')
    group.add_argument('--special-tokens-file', type=str, default=None,
                       help='Path to the BPE special tokens file.')
    parser.add_argument("--max-prompt-seq-len", type=int, default=256,
                        help="The maximum prompt length during RLHF Training.")
    group.add_argument('--mmap-warmup', action='store_true',
                       help='Warm up mmap files.')
    group.add_argument('--tokenizer-type', type=str,
                       default=None,
                       choices=['BertWordPieceLowerCase',
                                'BertWordPieceCase',
                                'GPT2BPETokenizer',
                                'SentencePieceTokenizer',
                                'GPTSentencePieceTokenizer',
                                'HFTokenizer',
                                'NullTokenizer',
                                'AquilaTokenizer',
                                'Llama2Tokenizer',
                                'Llama3Tokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--trust-remote-code', action='store_true', default=False,
                       help='To run HFTokenizer model from local path.')
    group.add_argument('--data-impl', type=str, default='infer',
                       choices=['mmap', 'infer'],
                       help='Implementation of indexed datasets.')
    group.add_argument('--train-data-exact-num-epochs', type=int, default=None,
                       help='When building the train dataset, force it to be '
                       'an exact number of epochs of the raw data')
    group.add_argument('--return-data-index', action='store_true',
                       help='Return the index of data sample.')
    group.add_argument('--data-efficiency-curriculum-learning', action='store_true',
                       help='Use DeepSpeed data efficiency library curriculum learning feature.')
    group.add_argument('--train-idx-path', type=str, default=None,
                       help='Force to use certain index file.')
    group.add_argument('--train-desc-path', type=str, default=None,
                       help='Force to use certain index file.')
    group.add_argument('--train-doc-idx-path', type=str, default=None,
                       help='Force to use certain index file.')
    group.add_argument('--train-sample-idx-path', type=str, default=None,
                       help='Force to use certain index file.')
    group.add_argument('--train-shuffle-idx-path', type=str, default=None,
                       help='Force to use certain index file.')
    group.add_argument('--repeated-dataloader', action='store_true',
                       help='Once all the data has been loaded, reuse the DataLoader.')
    return parser
    

def _add_zero_args(parser):
    """Text generate arguments."""

    group = parser.add_argument_group('ZeRO configurations', 'configurations')
    group.add_argument("--zero-stage", type=int, default=1.0)
    group.add_argument('--zero-reduce-scatter', action='store_true',
                       help='Use reduce scatter if specified')
    group.add_argument('--zero-contigious-gradients', action='store_true',
                       help='Use contigious memory optimizaiton if specified')
    group.add_argument("--zero-reduce-bucket-size", type=int, default=0.0)
    group.add_argument("--zero-allgather-bucket-size", type=int, default=0.0)
    group.add_argument('--remote-device', type=str, default='none', choices=['none', 'cpu', 'nvme'],
                      help='Remote device for ZeRO-3 initialized parameters.')
    group.add_argument('--use-pin-memory', action='store_true',
                     help='Use pinned CPU memory for ZeRO-3 initialized model parameters.')
    return parser

def _add_memoryopt_args(parser):
    """Memory optimization arguments."""

    group = parser.add_argument_group('Memory optimizations', 'configurations')
    group.add_argument("--scattered-embeddings", action='store_true',
                       help='Save memory by scattering embedding activations. '
                            'Introduces dropout differences across MP configurations.')
    group.add_argument("--split-transformers", action='store_true',
                       help='Save memory by splitting transformer layers into two parts, '
                       'allowing for more frequent activation checkpoint savings.')
    group.add_argument("--memory-centric-tiled-linear", action="store_true",
                       help='Save memory by tiling with deepspeed.zero.TiledLinear.')
    group.add_argument("--tile-factor", type=int, default=1,
                       help='Make all linear layers the same size of [hidden/tile_factor, hidden/tile_factor]. '
                            'Must be enabled with --memory-centric-tiled-linear. '
                            'Example A: if tile_factor=1, the qkv layer [hidden, 3* hidden] would be converted into [1,3] tiles of size [hidden,hidden]. '
                            'Example B: if tile_factor=2, the intermediate layer [4*hidden, hidden] will be converted into [8, 2] tiles of size [hidden/2, hidden/2]. '
                            'Default is 1.')

    return parser

def _add_activation_checkpoint_args(parser):
    group = parser.add_argument_group('Activation Checkpointing',
                                      'Checkpointing Configurations')
    group.add_argument('--deepspeed-activation-checkpointing', action='store_true',
                       help='uses activation checkpointing from deepspeed')
    group.add_argument('--partition-activations', action='store_true',
                       help='partition Activations across GPUs before checkpointing.')
    group.add_argument('--contigious-checkpointing', action='store_true',
                       help='Contigious memory checkpointing for activatoins.')
    group.add_argument('--checkpoint-in-cpu', action='store_true',
                       help='Move the activation checkpoints to CPU.')
    group.add_argument('--synchronize-each-layer', action='store_true',
                       help='does a synchronize at the beginning and end of each checkpointed layer.')
    group.add_argument('--profile-backward', action='store_true',
                       help='Enables backward pass profiling for checkpointed layers.')
    return parser

def _add_retro_args(parser):
    group = parser.add_argument_group(title='retro')
    group.add_argument('--retro-workdir', default=None,
                       help='Retro working directory, which contains the '
                       'preprocessed data for for pretraining. This directory '
                       'is built during preprocessing (see '
                       'tools/retro/README.md), and contains subdirectories '
                       'for the chunk database and pretraining neighbors.')
    group.add_argument("--retro-return-doc-ids", action="store_true",
                       help="Turn this on when preprocessing retro data.")
    
    # Enforce argument naming convention.
    for action in group._group_actions:
        prefix = action.dest.split("_")[0]
        assert prefix == "retro", \
            "Retro args must be prefixed with '--retro-*', for consistent " \
            "styling. Please fix '%s'." % ", ".join(action.option_strings)

    return parser

def validate_args(args, defaults={}):

    # Temporary
    assert args.non_persistent_ckpt_type in ['global', 'local', None], \
        'Currently only global and local checkpoints are supported'
    if args.non_persistent_ckpt_type == 'local':
        try:
            from nvidia_resiliency_ext.checkpointing.local.ckpt_managers.local_manager import \
                LocalCheckpointManager
        except ModuleNotFoundError as e:
            raise RuntimeError('nvidia_resiliency_ext is required for local checkpointing') from e

    # Load saved args from Retro (if applicable).
    load_retro_args(args)

    # Set args.use_dist_ckpt from args.ckpt_format.
    if args.use_legacy_models:
        assert args.ckpt_format == "torch", \
            "legacy model format only supports the 'torch' checkpoint format."
    update_use_dist_ckpt(args)

    if args.encoder_pipeline_model_parallel_size == 0 and args.num_experts == 0:
        assert args.encoder_tensor_model_parallel_size == args.tensor_model_parallel_size,  "If non-MOE encoder shares first decoder pipeline rank it must have the same TP as the decoder."

    if args.encoder_tensor_model_parallel_size > 0:
        assert args.num_attention_heads % args.encoder_tensor_model_parallel_size == 0
        assert args.encoder_tensor_model_parallel_size <= args.tensor_model_parallel_size, "We do not support encoders with more TP than the decoder."

    if args.encoder_pipeline_model_parallel_size > 0 and args.encoder_tensor_model_parallel_size == 0:
        args.encoder_tensor_model_parallel_size = args.tensor_model_parallel_size

    if args.deepspeed:
        encoder_model_size = args.encoder_tensor_model_parallel_size * args.encoder_pipeline_model_parallel_size * args.ds_sequence_parallel_size
        decoder_model_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.ds_sequence_parallel_size
    else:
        encoder_model_size = args.encoder_tensor_model_parallel_size * args.encoder_pipeline_model_parallel_size * args.context_parallel_size
        decoder_model_size = args.tensor_model_parallel_size * args.pipeline_model_parallel_size * args.context_parallel_size
    total_model_size = encoder_model_size + decoder_model_size

    # Total model size.
    assert args.world_size % total_model_size == 0, (
        f"world size ({args.world_size}) is not divisible by total_model_size ({encoder_model_size=} + {decoder_model_size=})"
    )

    if args.attention_backend == AttnBackend.local:
        assert args.spec[0] == 'local' , '--attention-backend local is only supported with --spec local'

    # Zero bubble pipeline is defined on deepspeed's scheduler
    if args.enable_zbh1_pipeline:
        assert args.deepspeed, 'Use DeepSpeed to use zero-bubble H1 pipeline'
        assert args.sequence_parallel == False, "Sequence Parallel not tested, proceed at own will by removing this line"
    if args.enable_zbh1_exact_semantics:
        assert args.enable_zbh1_pipeline, 'Exact semantics require ZBH1 pipeline enabled'
    # Pipeline model parallel size.
    args.transformer_pipeline_model_parallel_size = args.pipeline_model_parallel_size

    args.data_parallel_size = args.world_size // total_model_size

    if args.rank == 0:
        if args.deepspeed:
            print('using world size: {}, data-parallel size: {}, '
              'deepspeed sequence-parallel size: {}, '
              'hierarchical context-parallel sizes: {}'
              'tensor-model-parallel size: {}, '
              'encoder-tensor-model-parallel size: {}, '
              'pipeline-model-parallel size: {}, '
              'encoder-pipeline-model-parallel size: {}'.format(
                  args.world_size, args.data_parallel_size,
                  args.context_parallel_size,
                  args.hierarchical_context_parallel_sizes,
                  args.tensor_model_parallel_size,
                  args.encoder_tensor_model_parallel_size,
                  args.pipeline_model_parallel_size,
                  args.encoder_pipeline_model_parallel_size), flush=True) 
        else:
            print('using world size: {}, data-parallel size: {}, '
                'context-parallel size: {}, '
                'tensor-model-parallel size: {}, '
                'encoder-tensor-model-parallel size: {}, '
                'pipeline-model-parallel size: {}, '
                'encoder-pipeline-model-parallel size: {}'.format(
                    args.world_size, args.data_parallel_size,
                    args.ds_sequence_parallel_size,
                    args.tensor_model_parallel_size,
                    args.encoder_tensor_model_parallel_size,
                    args.pipeline_model_parallel_size,
                    args.encoder_pipeline_model_parallel_size), flush=True)

    # Checks.
    # if args.no_pipeline_parallel:
    #     assert args.pipeline_model_parallel_size == 1, \
    #         "pipeline_model_parallel_size must be 1 if pipeline parallel is disabled"
        
    if args.ds_sequence_parallel_size > 1:
        assert args.deepspeed, "deepspeed must be enable when ds_sequence_parallel_size > 1"
        assert args.context_parallel_size <= 1, "Megatron-lm CP is not compatible with Deppspeed SP"
        assert version.parse(deepspeed.__version__) >= version.parse("0.10.2"), "sequence parallelism requires DeepSpeed version 0.10.2+"

    # Backwards compatibility.
    if args.pipeline_model_parallel_split_rank is not None:
        args.encoder_pipeline_model_parallel_size = args.pipeline_model_parallel_split_rank
        args.pipeline_model_parallel_size -= args.encoder_pipeline_model_parallel_size
        assert args.pipeline_model_parallel_size > 0

    if args.hierarchical_context_parallel_sizes:
        from numpy import prod
        assert args.context_parallel_size == prod(args.hierarchical_context_parallel_sizes)
    if "a2a+p2p" in args.cp_comm_type:
        assert args.hierarchical_context_parallel_sizes is not None, \
        "--hierarchical-context-parallel-sizes must be set when a2a+p2p is used in cp comm"

    if args.expert_tensor_parallel_size is None:
        args.expert_tensor_parallel_size = args.tensor_model_parallel_size

    # Deprecated arguments.
    assert args.batch_size is None, '--batch-size argument is no longer ' \
        'valid, use --micro-batch-size instead'
    del args.batch_size
    assert args.warmup is None, '--warmup argument is no longer valid, use ' \
        '--lr-warmup-fraction instead'
    del args.warmup
    assert args.model_parallel_size is None, '--model-parallel-size is no ' \
        'longer valid, use --tensor-model-parallel-size instead'
    del args.model_parallel_size

    # HACK: below is commented because DeepSpeed still relies on the old
    # activation checkpointing mechanism.
    # if args.checkpoint_activations:
    #     if args.rank == 0:
    #         print('--checkpoint-activations is no longer valid, use --recompute-activations, '
    #               'or, for more control, --recompute-granularity and --recompute-method.')
    #     exit()
    # del args.checkpoint_activations

    if args.recompute_activations:
        args.recompute_granularity = 'selective'
    del args.recompute_activations

    # Set input defaults.
    for key in defaults:
        # For default to be valid, it should not be provided in the
        # arguments that are passed to the program. We check this by
        # ensuring the arg is set to None.
        if getattr(args, key, None) is not None:
            if args.rank == 0:
                print('WARNING: overriding default arguments for {key}:{v} \
                       with {key}:{v2}'.format(key=key, v=defaults[key],
                                               v2=getattr(args, key)),
                                               flush=True)
        else:
            setattr(args, key, defaults[key])

    if args.data_path is not None and args.split is None:
        legacy_default_split_value = '969, 30, 1'
        if args.rank == 0:
            print('WARNING: Please specify --split when using --data-path. Using legacy default value '
                  f'of "{legacy_default_split_value}"')
        args.split = legacy_default_split_value

    use_data_path = (args.data_path is not None) or (args.data_args_path is not None)
    if use_data_path:
        # Exactly one of the two has to be None if we use it.
        assert (args.data_path is None) or (args.data_args_path is None)
    use_per_split_data_path = any(
        elt is not None
        for elt in [args.train_data_path, args.valid_data_path, args.test_data_path]) or \
            args.per_split_data_args_path is not None
    if use_per_split_data_path:
         # Exactly one of the two has to be None if we use it.
        assert any(elt is not None
                   for elt in [args.train_data_path, args.valid_data_path, args.test_data_path]) is False or \
            args.per_split_data_args_path is None

    # Batch size.
    assert args.micro_batch_size is not None
    assert args.micro_batch_size > 0
    if args.global_batch_size is None:
        args.global_batch_size = args.micro_batch_size * args.data_parallel_size
        if args.rank == 0:
            print('setting global batch size to {}'.format(
                args.global_batch_size), flush=True)
    assert args.global_batch_size > 0
    
    # Uneven virtual pipeline parallelism
    assert args.num_layers_per_virtual_pipeline_stage is None or args.num_virtual_stages_per_pipeline_rank is None, \
        '--num-layers-per-virtual-pipeline-stage and --num-virtual-stages-per-pipeline-rank cannot be set at the same time'
                  
    if args.num_layers_per_virtual_pipeline_stage is not None or args.num_virtual_stages_per_pipeline_rank is not None:
        if args.overlap_p2p_comm:
            assert args.pipeline_model_parallel_size > 1, \
                'When interleaved schedule is used, pipeline-model-parallel size '\
                'should be greater than 1'
        else:
            assert args.pipeline_model_parallel_size > 2, \
                'When interleaved schedule is used and p2p communication overlap is disabled, '\
                'pipeline-model-parallel size should be greater than 2 to avoid having multiple '\
                'p2p sends and recvs between same 2 ranks per communication batch'
        
        if args.num_virtual_stages_per_pipeline_rank is None:
            assert args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None, \
                'please use --num-virtual-stages-per-pipeline-rank to specify virtual pipeline parallel degree when enable uneven pipeline parallelism'
            num_layers = args.num_layers
            
            if args.account_for_embedding_in_pipeline_split:
                num_layers += 1
            
            if args.account_for_loss_in_pipeline_split:
                num_layers += 1
            
            assert num_layers % args.transformer_pipeline_model_parallel_size == 0, \
                'number of layers of the model must be divisible pipeline model parallel size'
            num_layers_per_pipeline_stage = num_layers // args.transformer_pipeline_model_parallel_size
            
            assert num_layers_per_pipeline_stage % args.num_layers_per_virtual_pipeline_stage == 0, \
                'number of layers per pipeline stage must be divisible number of layers per virtual pipeline stage'
            args.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
                args.num_layers_per_virtual_pipeline_stage
        else:
            args.virtual_pipeline_model_parallel_size = args.num_virtual_stages_per_pipeline_rank
        if args.num_layers_per_stage is None:
            assert args.num_layers is not None
            # Double check divisibility check here since check above is if guarded.
            assert args.num_layers % args.transformer_pipeline_model_parallel_size == 0, \
                'Number of layers should be divisible by the pipeline-model-parallel size'
            num_layers_per_pipeline_stage = args.num_layers // args.transformer_pipeline_model_parallel_size
            assert num_layers_per_pipeline_stage % args.num_layers_per_virtual_pipeline_stage == 0, \
                'Number of layers per pipeline stage must be divisible by number of layers per virtual pipeline stage'
            args.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
                args.num_layers_per_virtual_pipeline_stage
        else:
            stage_split = args.num_layers_per_stage[::2]
            num_layers_per_stage_split = args.num_layers_per_stage[1::2]
            num_layers_per_stage = []
            for i in range(len(stage_split)):
                for j in range(stage_split[i]):
                    num_layers_per_stage.append(num_layers_per_stage_split[i])
            args.num_layers_per_stage = num_layers_per_stage
            total_virtual_pipeline_stage_num = len(args.num_layers_per_stage)
            assert total_virtual_pipeline_stage_num % args.pipeline_model_parallel_size == 0, \
                'len(args.num_layers_per_stage) is not divisible by pp size'
            args.virtual_pipeline_model_parallel_size = len(args.num_layers_per_stage) // \
                args.pipeline_model_parallel_size
    else:
        if args.num_layers_per_stage is not None:
            stage_split = args.num_layers_per_stage[::2]
            num_layers_per_stage_split = args.num_layers_per_stage[1::2]
            num_layers_per_stage = []
            for i in range(len(stage_split)):
                for j in range(stage_split[i]):
                    num_layers_per_stage.append(num_layers_per_stage_split[i])
            args.num_layers_per_stage = num_layers_per_stage
            assert len(args.num_layers_per_stage) == args.pipeline_model_parallel_size, \
                'len(args.num_layers_per_stage) do not match with pp size'
        args.virtual_pipeline_model_parallel_size = None
        # Overlap P2P communication is disabled if not using the interleaved schedule.
        args.overlap_p2p_comm = False
        args.align_param_gather = False
        # Only print warning if PP size > 1.
        if args.rank == 0 and args.pipeline_model_parallel_size > 1:
            print('WARNING: Setting args.overlap_p2p_comm and args.align_param_gather to False '
                  'since non-interleaved schedule does not support overlapping p2p communication '
                  'and aligned param AG')

        if args.decoder_first_pipeline_num_layers is None and args.decoder_last_pipeline_num_layers is None:
            # Divisibility check not applicable for T5 models which specify encoder_num_layers
            # and decoder_num_layers.
            if args.num_layers is not None:
                num_layers = args.num_layers
                
                if args.account_for_embedding_in_pipeline_split:
                    num_layers += 1
                
                if args.account_for_loss_in_pipeline_split:
                    num_layers += 1
                    
                if args.num_layers_per_stage is None and args.custom_partition is None:
                    assert num_layers % args.transformer_pipeline_model_parallel_size == 0, \
                        'Number of layers should be divisible by the pipeline-model-parallel size'

    print("Virtual model parallel size ", args.virtual_pipeline_model_parallel_size)
    if args.overlap_param_gather:
        assert args.use_distributed_optimizer, \
            '--overlap-param-gather only supported with distributed optimizer'
        assert args.overlap_grad_reduce, \
            'Must use --overlap-param-gather with --overlap-grad-reduce'
        assert not args.use_legacy_models, \
            '--overlap-param-gather only supported with MCore models'

    ## RLHF Batch size check
    if args.RLHF:
        assert args.global_batch_size == args.micro_batch_size * args.data_parallel_size, \
            f"error with batch size setting. GBS should equal to MBS * DP"

    if getattr(args, "use_torch_fsdp2", False):
        assert is_torch_min_version("2.4.0"), \
            'FSDP2 requires PyTorch >= 2.4.0 with FSDP 2 support.'
        assert args.pipeline_model_parallel_size == 1, \
            '--use-torch-fsdp2 is not supported with pipeline parallelism'
        assert args.expert_model_parallel_size == 1, \
            '--use-torch-fsdp2 is not supported with expert parallelism'
        assert not args.use_distributed_optimizer, \
            "--use-torch-fsdp2 is not supported with MCore's distributed optimizer"
        assert not args.gradient_accumulation_fusion, \
            '--use-torch-fsdp2 is not supported with gradient accumulation fusion'
        assert args.ckpt_format == 'torch_dist', \
            '--use-torch-fsdp2 requires --ckpt-format torch_dist'
        assert args.untie_embeddings_and_output_weights, \
            '--use-torch-fsdp2 requires --untie-embeddings-and-output-weights'
        assert not args.fp16, \
            '--use-torch-fsdp2 not supported with fp16 yet'

    if args.overlap_param_gather_with_optimizer_step:
        assert args.use_distributed_optimizer, \
            '--overlap-param-gather-with-optimizer-step only supported with distributed optimizer'
        assert args.overlap_param_gather, \
            'Must use --overlap-param-gather-with-optimizer-step with --overlap-param-gather'
        assert args.virtual_pipeline_model_parallel_size is not None, \
            '--overlap-param-gather-with-optimizer-step only supported with interleaved pipeline parallelism'
        assert not args.use_dist_ckpt, \
            '--overlap-param-gather-with-optimizer-step not supported with distributed checkpointing yet'

    dtype_map = {
        'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp8': torch.uint8,
    }
    map_dtype = lambda d: d if isinstance(d, torch.dtype) else dtype_map[d]

    args.main_grads_dtype = map_dtype(args.main_grads_dtype)
    args.main_params_dtype = map_dtype(args.main_params_dtype)
    args.exp_avg_dtype = map_dtype(args.exp_avg_dtype)
    args.exp_avg_sq_dtype = map_dtype(args.exp_avg_sq_dtype)

    if args.fp8_param_gather:
        assert args.use_distributed_optimizer, \
            '--fp8-param-gather only supported with distributed optimizer'

    # Parameters dtype.
    args.params_dtype = torch.float
    if args.fp16:
        assert not args.bf16
        args.params_dtype = torch.half
        # Turn off checking for NaNs in loss and grads if using dynamic loss scaling,
        # where NaNs in grads / loss are signal to the loss scaler.
        if not args.loss_scale:
            args.check_for_nan_in_loss_and_grad = False
            if args.rank == 0:
                print('WARNING: Setting args.check_for_nan_in_loss_and_grad to False since '
                      'dynamic loss scaling is being used')
    if args.bf16:
        assert not args.fp16
        args.params_dtype = torch.bfloat16
        # bfloat16 requires gradient accumulation and all-reduce to
        # be done in fp32.

        if args.accumulate_allreduce_grads_in_fp32:
            assert args.main_grads_dtype == torch.float32, \
                "--main-grads-dtype can only be fp32 when --accumulate-allreduce-grads-in-fp32 is set"

        if not args.accumulate_allreduce_grads_in_fp32 and args.main_grads_dtype == torch.float32:
            args.accumulate_allreduce_grads_in_fp32 = True
            if args.rank == 0:
                print('accumulate and all-reduce gradients in fp32 for '
                      'bfloat16 data type.', flush=True)

    if args.rank == 0:
        print('using {} for parameters ...'.format(args.params_dtype),
              flush=True)

    if args.dataloader_type is None:
        args.dataloader_type = 'single'

    # data
    assert args.num_dataset_builder_threads > 0

    # Consumed tokens.
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0
    args.consumed_train_tokens = 0

    # Support for variable sequence lengths across batches/microbatches.
    # set it if the dataloader supports generation of variable sequence lengths
    # across batches/microbatches. Due to additional communication overhead
    # during pipeline parallelism, it should not be set if sequence length
    # is constant during training.
    # args.variable_seq_lengths = True

    # Iteration-based training.
    if args.train_iters:
        # If we use iteration-based training, make sure the
        # sample-based options are off.
        assert args.train_samples is None, \
            'expected iteration-based training'
        assert args.lr_decay_samples is None, \
            'expected iteration-based learning rate decay'
        assert args.lr_warmup_samples == 0, \
            'expected iteration-based learning rate warmup'
        assert args.rampup_batch_size is None, \
            'expected no batch-size rampup for iteration-based training'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_iters == 0, \
                'can only specify one of lr-warmup-fraction and lr-warmup-iters'

    # Sample-based training.
    if args.train_samples:
        # If we use sample-based training, make sure the
        # iteration-based options are off.
        assert args.train_iters is None, \
            'expected sample-based training'
        assert args.lr_decay_iters is None, \
            'expected sample-based learning rate decay'
        assert args.lr_warmup_iters == 0, \
            'expected sample-based learnig rate warmup'
        if args.lr_warmup_fraction is not None:
            assert args.lr_warmup_samples == 0, \
                'can only specify one of lr-warmup-fraction ' \
                'and lr-warmup-samples'

    if args.num_layers is not None:
        assert args.encoder_num_layers is None, \
            'cannot have both num-layers and encoder-num-layers specified'
        args.encoder_num_layers = args.num_layers
    else:
        if not args.use_dataset_only:
            assert args.encoder_num_layers is not None, \
                'either num-layers or encoder-num-layers should be specified'
            args.num_layers = args.encoder_num_layers

    # Check required arguments.
    if not args.use_dataset_only:
        required_args = ['num_layers', 'hidden_size', 'num_attention_heads',
                        'max_position_embeddings']
        for req_arg in required_args:
            _check_arg_is_not_none(args, req_arg)

    # Checks.
    if not args.use_dataset_only:
        if args.ffn_hidden_size is None:
            if args.swiglu:
                # reduce the dimnesion for MLP since projections happens on
                # two linear layers. this keeps the number of paramters in
                # the same ballpark as the counterpart with 4*h size
                # we keep it a multiple of 64, which means the actual tensor size
                # will be a multiple of 64 / tp_size
                args.ffn_hidden_size = int((4 * args.hidden_size * 2 / 3) / 64) * 64
            else:
                args.ffn_hidden_size = 4 * args.hidden_size

        if args.kv_channels is None:
            assert args.hidden_size % args.num_attention_heads == 0
            args.kv_channels = args.hidden_size // args.num_attention_heads

    if args.seq_length is not None and args.context_parallel_size > 1:
        assert args.seq_length % (args.context_parallel_size * 2) == 0, \
            'seq-length should be a multiple of 2 * context-parallel-size ' \
            'if context-parallel-size > 1.'

    if args.seq_length is not None:
        assert args.encoder_seq_length is None
        args.encoder_seq_length = args.seq_length
    else:
        assert args.encoder_seq_length is not None
        args.seq_length = args.encoder_seq_length

    if not args.use_dataset_only:
        if args.seq_length is not None:
            assert args.max_position_embeddings >= args.seq_length, \
                f"max_position_embeddings ({args.max_position_embeddings}) must be greater than " \
                f"or equal to seq_length ({args.seq_length})."
        if args.decoder_seq_length is not None:
            assert args.max_position_embeddings >= args.decoder_seq_length
    # When rotary position embeddings is used, set add_position_embedding
    # to false to turn off absolute position embedding.
    if args.use_rotary_position_embeddings:
        args.add_position_embedding = False
    if args.lr is not None:
        assert args.min_lr <= args.lr
    if args.save is not None:
        assert args.save_interval is not None
    # Mixed precision checks.
    if args.fp16_lm_cross_entropy:
        assert args.fp16, 'lm cross entropy in fp16 only support in fp16 mode.'
    if args.fp32_residual_connection:
        assert args.fp16 or args.bf16, \
            'residual connection in fp32 only supported when using fp16 or bf16.'

    if args.moe_grouped_gemm:
        assert args.bf16, 'Currently GroupedGEMM for MoE only supports bf16 dtype.'
        #dc = torch.cuda.get_device_capability()
        #assert dc[0] >= 8, "Unsupported compute capability for GroupedGEMM kernels."

    assert not (args.moe_block_sparse_gemm and args.moe_grouped_gemm), \
        'moe_block_sparse_gemm and moe_grouped_gemm cannot be used together.'

    if not args.use_dataset_only:
        if args.weight_decay_incr_style == 'constant':
            assert args.start_weight_decay is None
            assert args.end_weight_decay is None
            args.start_weight_decay = args.weight_decay
            args.end_weight_decay = args.weight_decay
        else:
            assert args.start_weight_decay is not None
            assert args.end_weight_decay is not None

    # Persistent fused layer norm.
    if not is_torch_min_version("1.11.0a0"):
        args.no_persist_layer_norm = True
        if args.rank == 0:
            print('Persistent fused layer norm kernel is supported from '
                  'pytorch v1.11 (nvidia pytorch container paired with v1.11). '
                  'Defaulting to no_persist_layer_norm=True')

    # Activation checkpointing.
    if args.distribute_checkpointed_activations:
        assert args.checkpoint_activations, \
            'for distribute-checkpointed-activations to work you '\
            'need to enable checkpoint-activations'

    # Activation recomputing.
    if args.distribute_saved_activations:
        assert args.tensor_model_parallel_size > 1, 'can distribute ' \
            'recomputed activations only across tensor model ' \
            'parallel groups'
        assert args.recompute_granularity == 'full', \
            'distributed recompute activations is only '\
            'application to full recompute granularity'
        assert args.recompute_method is not None, \
            'for distributed recompute activations to work you '\
            'need to use a recompute method '
        assert is_torch_min_version("1.10.0a0"), \
            'distributed recompute activations are supported for pytorch ' \
            'v1.10 and above (Nvidia Pytorch container >= 21.07). Current ' \
            f'pytorch version is v{get_torch_version()}.'

    if args.recompute_granularity == 'selective':
        assert args.recompute_method is None, \
            'recompute method is not yet supported for ' \
            'selective recomputing granularity'

    if args.recompute_num_layers_per_stage is not None:
        assert args.recompute_granularity == 'full', \
            'recompute-num-layers-per-stage is only'\
            'application to full recompute granularity'
        assert args.recompute_method_per_stage is not None, \
            'recompute_method_per_stage must be used with '\
            'recompute_num_layers_per_stage '

        recompute_num_layers_stage_split = args.recompute_num_layers_per_stage[::2]
        recompute_num_layers_layer_split = args.recompute_num_layers_per_stage[1::2]
        recompute_methods_stage_split = args.recompute_method_per_stage[::2]
        recompute_methods_method_split = args.recompute_method_per_stage[1::2]

        assert len(recompute_num_layers_stage_split) == len(recompute_num_layers_layer_split), \
            'args.recompute_num_layers_per_stage setting must match form: n0, layers0, n1, layers1, ...'
        assert len(recompute_methods_stage_split) == len(recompute_methods_method_split), \
            'args.recompute_method_per_stage setting must match form: n0, layers0, n1, layers1, ...'
        if args.virtual_pipeline_model_parallel_size != None:
            assert args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size == sum(recompute_num_layers_stage_split), \
                'args.recompute_num_layers_per_stage setting:' \
                'the sum of n0, n1, ... should be equal to pipeline-model-parallel-size * virtual_pipeline_model_parallel_size'
            assert args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size == sum(recompute_methods_stage_split), \
                'args.recompute_method_per_stage setting:' \
                'the sum of n0, n1, ... should be equal to pipeline-model-parallel-size * virtual_pipeline_model_parallel_size'
        else:
            assert args.pipeline_model_parallel_size == sum(recompute_num_layers_stage_split), \
                'args.recompute_num_layers_per_stage setting:' \
                'the sum of n0, n1, ... should be equal to pipeline-model-parallel-size.'
            assert args.pipeline_model_parallel_size == sum(recompute_methods_stage_split), \
                'args.recompute_method_per_stage setting:' \
                'the sum of n0, n1, ... should be equal to pipeline-model-parallel-size.'

        recompute_num_layers_per_stage = []
        for i in range(len(recompute_num_layers_stage_split)):
            for j in range(recompute_num_layers_stage_split[i]):
                recompute_num_layers_per_stage.append(recompute_num_layers_layer_split[i])
        recompute_method_per_stage = []
        for i in range(len(recompute_methods_stage_split)):
            for j in range(recompute_methods_stage_split[i]):
                recompute_method_per_stage.append(recompute_methods_method_split[i])

        args.recompute_num_layers_per_stage = recompute_num_layers_per_stage
        args.recompute_method_per_stage = recompute_method_per_stage

    if args.custom_recompute_layers_per_stage:
        if args.virtual_pipeline_model_parallel_size is not None:
            assert len(args.custom_recompute_layers_per_stage) == args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size, \
                f"custom recompute_num_layers_per_stage length ({len(args.custom_recompute_layers_per_stage)}) should equal to total virtual pp stage size ({args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size})"
        else:
            assert len(args.custom_recompute_layers_per_stage) == args.pipeline_model_parallel_size, \
                f"custom recompute_num_layers_per_stage ({len(args.custom_recompute_layers_per_stage)}) length should equal to PP size ({args.pipeline_model_parallel_size})"
        
        ## 若是deepseed使用自定义重计算pp stage则不考虑如下
        if not args.deepspeed:
            assert args.recompute_granularity == 'full', \
            'custom recompute layers pp stage is only '\
            'application to full recompute granularity'
        
            if args.virtual_pipeline_model_parallel_size is None:
                num_layers_per_stage = args.num_layers // args.pipeline_model_parallel_size
            else:
                num_layers_per_stage = args.num_layers_per_virtual_pipeline_stage
            if args.custom_partition is None:
                assert max(args.custom_recompute_layers_per_stage) <= num_layers_per_stage, \
                "recompute layers per PP stage should small than num layers per stage." \
                f"get max recompute layers: {max(args.custom_recompute_layers_per_stage)}" \
                f"average num layers per stage: {num_layers_per_stage}"
            else:
                for i in range(args.pipeline_model_parallel_size):
                    assert args.custom_recompute_layers_per_stage[i] <= args.custom_partition[i], \
                    "recompute layers per PP stage should small the num layers of PP stage" \
                    f"stage ({i}): recompute layers ({args.custom_recompute_layers_per_stage[i]})  >  stage layers ({args.custom_partition[i]})"

    if args.recompute_num_layers_per_stage is None and args.custom_recompute_layers_per_stage:
        args.recompute_num_layers_per_stage = args.custom_recompute_layers_per_stage
    elif args.recompute_num_layers_per_stage is not None and args.custom_recompute_layers_per_stage is None:
        args.custom_recompute_layers_per_stage = args.recompute_num_layers_per_stage

    if args.num_layers_per_stage is None and args.custom_partition:
        args.num_layers_per_stage = args.custom_partition
    elif args.num_layers_per_stage is not None and args.custom_partition is None:
        args.custom_partition = args.num_layers_per_stage

    # disable sequence parallelism when tp=1
    # to avoid change in numerics when
    # sequence_parallelism is enabled.
    if args.tensor_model_parallel_size == 1:
        if args.sequence_parallel:
            warnings.warn("Disabling sequence parallelism because tensor model parallelism is disabled")
        args.sequence_parallel = False

    if args.tp_comm_overlap:
        assert args.sequence_parallel == True, 'Tensor parallel communication/GEMM overlap can happen only when sequence parallelism is enabled'

    # disable async_tensor_model_parallel_allreduce when
    # model parallel memory optimization is enabled
    if args.sequence_parallel:
        args.async_tensor_model_parallel_allreduce = False
        if getattr(args, "use_torch_fsdp2", False):
            warnings.warn(
                "Using sequence parallelism with FSDP2 together. Try not to using them "
                "together since they require different CUDA_MAX_CONNECTIONS settings "
                "for best performance. sequence parallelism requires setting the "
                "environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1 while FSDP2 "
                "requires not setting CUDA_DEVICE_MAX_CONNECTIONS=1 for better parallelization.")

    # TODO: currently DeepSpeed seems to be incompatible with
    # async_tensor_model_parallel_allreduce thus temporarily disabling it.
    # Need further investigation.
    if args.deepspeed:
        args.async_tensor_model_parallel_allreduce = False

    if not args.use_dataset_only:
        if os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1":
            if args.sequence_parallel:
                raise RuntimeError(
                    "Using sequence parallelism requires setting the environment variable "
                    "CUDA_DEVICE_MAX_CONNECTIONS to 1")
            if args.async_tensor_model_parallel_allreduce:
                raise RuntimeError(
                    "Using async gradient all reduce requires setting the environment "
                    "variable CUDA_DEVICE_MAX_CONNECTIONS to 1")

    # Disable bias gelu fusion if we are disabling bias altogether
    if not args.add_bias_linear:
        args.bias_gelu_fusion = False

    # Keep the 'add bias' args in sync; add_qkv_bias is more targeted.
    if args.add_bias_linear:
        args.add_qkv_bias = True

    # Retro checks.
    if args.retro_add_retriever:

        # Train samples should be auto-loaded.
        assert args.train_samples is not None, \
            "args.train_samples should be auto-loaded from the retro config."

        # Sequence parallelism unsupported.
        assert not args.sequence_parallel, \
            "retro currently does not support sequence parallelism."

        # Pipeline parallelism unsupported.
        assert args.pipeline_model_parallel_size == 1, \
            "retro currently does not support pipeline parallelism."

    if args.decoupled_lr is not None or args.decoupled_min_lr is not None:
        assert not args.use_legacy_models, \
            '--decoupled-lr and --decoupled-min-lr is not supported in legacy models.'

    ## meg-ds start
    args.curriculum_learning_legacy = False
    args.compression_training = False
    args.mos = False
    args.kd = False

    # FlashAttention
    args.use_flash_attn = args.use_flash_attn or args.use_flash_attn_v1 or args.use_flash_attn_triton or args.use_flash_attn_v2

    # AML
    if args.aml_data_download_path is not None:
        data_paths = []
        for path in args.data_path:
            data_paths.append(f"{args.aml_data_download_path}/{path}")
        args.data_path = data_paths

    # GQA
    if not args.use_dataset_only:
        if args.group_query_attention:
            args.num_key_value_heads = args.num_query_groups
        if args.num_key_value_heads is None:
            args.num_key_value_heads = args.num_attention_heads
        assert args.num_attention_heads % args.num_key_value_heads == 0, \
            f"num_attention_heads must be divisible by num_key_value_heads (got `num_attention_heads`: {args.num_attention_heads} " \
            f"and `num_key_value_heads`: {args.num_key_value_heads})."
    ## meg-ds end

    # Legacy RoPE arguments
    if args.use_rotary_position_embeddings:
        args.position_embedding_type = 'rope'
    if args.rotary_interleaved and args.apply_rope_fusion:
        raise RuntimeError('--rotary-interleaved does not work with rope_fusion.')
    if args.rotary_interleaved and args.use_legacy_models:
        raise RuntimeError('--rotary-interleaved is not supported in legacy models.')
    if args.position_embedding_type != 'rope':
        args.apply_rope_fusion = False

    # Would just need to add 'NoPE' as a position_embedding_type to support this, but for now
    # don't allow it to keep things simple
    if not args.add_position_embedding and args.position_embedding_type != 'rope':
        raise RuntimeError('--no-position-embedding is deprecated, use --position-embedding-type')

    # Relative position embeddings arguments
    if args.position_embedding_type == 'relative':
        assert (
            args.transformer_impl == "transformer_engine"
        ), 'Local transformer implementation currently does not support attention bias-based position embeddings.'

    # MoE Spec check
    if args.num_experts == 0:
        args.num_experts = None
    if args.num_experts is not None:
        assert args.spec is None, "Model Spec must be None when using MoEs"

    if args.moe_ffn_hidden_size is None:
        args.moe_ffn_hidden_size = args.ffn_hidden_size

    # Context parallel
    # if args.context_parallel_size > 1:
    #     assert not args.use_legacy_models, "Context parallelism is not supported in legacy models."

    # Expert parallelism check
    if args.expert_model_parallel_size  > 1:
        assert args.num_experts is not None, "num_experts must be non None to use expert model parallelism"
        assert args.num_experts % args.expert_model_parallel_size == 0, \
            "Number of experts should be a multiple of expert model parallel_size."
        assert not args.fp16, \
            "Expert parallelism is not supported with fp16 training."

    # Distributed checkpointing checks
    if args.use_dist_ckpt and args.use_legacy_models:
        raise RuntimeError('--use-dist-ckpt is not supported in legacy models.')

    # Data blend checks
    assert args.mock_data + \
           bool(args.data_path) + \
           any([args.train_data_path, args.valid_data_path, args.test_data_path]) \
           <= 1, "A single data source must be provided in training mode, else None"

    # Deterministic mode
    if args.deterministic_mode:
        assert not args.use_flash_attn, "Flash attention can not be used in deterministic mode."
        assert not args.cross_entropy_loss_fusion, "Cross Entropy Fusion is currently not deterministic."

        all_reduce_choices = ["Tree", "Ring", "CollnetDirect", "CollnetChain", "^NVLS"]
        assert os.getenv("NCCL_ALGO", -1) != -1 and os.getenv("NCCL_ALGO") in all_reduce_choices, \
            f"NCCL_ALGO must be one of {all_reduce_choices}."

        torch.use_deterministic_algorithms(True)

    # Update the printed args to reflect that `apply_query_key_layer_scaling` also controls `attention_softmax_in_fp32`
    if args.apply_query_key_layer_scaling:
        args.attention_softmax_in_fp32 = True

    # Checkpointing
    if args.ckpt_fully_parallel_save_deprecated and args.rank == 0:
        print('--ckpt-fully-parallel-save flag is deprecated and has no effect.'
              ' Use --no-ckpt-fully-parallel-save to disable parallel save.')
    if (
        args.use_dist_ckpt
        and not args.ckpt_fully_parallel_save
        and args.use_distributed_optimizer
        and args.rank == 0
    ):
        print('Warning: With non-parallel ckpt save and DistributedOptimizer,'
              ' it will be impossible to resume training with different parallelism.'
              ' Consider removing flag --no-ckpt-fully-parallel-save.')
    if args.use_dist_ckpt_deprecated and args.rank == 0:
        print('--use-dist-ckpt is deprecated and has no effect.'
              ' Use --ckpt-format to select the checkpoint format.')
    if args.dist_ckpt_format_deprecated and args.rank == 0:
        print('--dist-ckpt-format is deprecated and has no effect.'
              ' Use --ckpt-format to select the checkpoint format.')

    # Inference args
    if args.inference_batch_times_seqlen_threshold > -1:
        assert args.pipeline_model_parallel_size > 1, \
            "--inference-batch-times-seqlen-threshold requires setting --pipeline-model-parallel-size > 1."

    # MoE upcycling check
    if args.moe_use_upcycling:
        assert args.save is not None, "When using upcycling, the --save option must be specified."
        if not args.no_load_optim:
            args.no_load_optim = True
            print('Warning: disabling --no-load-optim for upcycling.')
        if not args.no_load_rng:
            args.no_load_rng = True
            print('Warning: disabling --no-load-rng for upcycling.')

    # MoE loss and include embedding and loss layer check
    if args.num_experts is not None:
        if args.moe_router_load_balancing_type != "none" or args.moe_z_loss_coeff is not None:
            assert not args.account_for_embedding_in_pipeline_split, \
                "Cannot support load balancing loss and z loss with --account-for-embedding-in-pipeline-split"
            assert not args.account_for_loss_in_pipeline_split, \
                "Cannot support load balancing loss and z loss with --account-for-loss-in-pipeline-split"
                

    if args.non_persistent_ckpt_type == "local":
        assert args.non_persistent_local_ckpt_dir is not None, "Tried to use local checkpointing without specifying --local-ckpt-dir!"
    if args.replication:
        assert args.replication_jump is not None, "--replication requires the value of --replication-jump!"
        assert args.non_persistent_ckpt_type == "local", f"--replication requires args.non_persistent_ckpt_type == 'local', but got: {args.non_persistent_ckpt_type}"
    elif args.replication_jump:
        print("Warning: --replication-jump was specified despite not using replication. Ignoring.")
        args.replication_jump = None
    if args.two_stage_p2p and args.sequence_parallel:
        raise RuntimeError('--two-stage-p2p only support sequence_parallel off.')

    if args.create_attention_mask_in_dataloader and args.rank == 0:
        print('WARNING: create_attention_mask_in_dataloader is True, do you really need it?!')

    # Print arguments.
    _print_args("arguments", args)

    if args.pp_delay:
        if not args.overlap_p2p_comm:
            args.pp_delay = False

    return args
