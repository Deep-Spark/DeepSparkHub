'''
Summarizes a set of results.
'''

from __future__ import print_function

import argparse
import copy
import glob
import json
import os
import re
import sys
import itertools
import pandas as pd

from ..compliance_checker import mlp_compliance
from ..compliance_checker.mlp_compliance import usage_choices, rule_choices
from ..compliance_checker.mlp_parser import parse_file
from ..rcp_checker import rcp_checker

from ..benchmark_meta import get_allowed_benchmarks, get_result_file_counts


def _get_sub_folders(folder):
    sub_folders = [
        os.path.join(folder, sub_folder) for sub_folder in os.listdir(folder)
    ]
    return [
        sub_folder for sub_folder in sub_folders if os.path.isdir(sub_folder)
    ]


def _read_json_file(json_file):
    with open(json_file, 'r') as f:
        try:
            content = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError('ERROR: Could not decode JSON struct '
                                       'in {}: {}'.format(json_file, e))
    return content


def _pretty_system_name(system_desc):
    system_name = system_desc['system_name']
    if system_name == 'tpu-v3':
        chips = int(system_desc['accelerators_per_node']) * 2
        return 'TPUv3.{}'.format(chips)
    return system_name


def _linkable_system_name(system_desc):
    system_name = system_desc['system_name']
    if system_name == 'tpu-v3':
        chips = int(system_desc['accelerators_per_node']) * 2
        return 'tpu-v3-{}'.format(chips)
    return system_name


def _pretty_accelerator_model_name(system_desc):
    accelerator_model_name = system_desc['accelerator_model_name']
    if accelerator_model_name == 'tpu-v3':
        return 'TPUv3'
    return accelerator_model_name


def _pretty_framework(system_desc):
    framework = system_desc['framework']
    if 'TensorFlow' in framework:
        commit_hash = re.search(r' commit hash = .*', framework)
        if commit_hash:
            return framework.replace(commit_hash.group(0), '')
    return framework


def _benchmark_alias(benchmark):
    if benchmark == 'mask':
        return 'maskrcnn'
    return benchmark


def _ruleset_url_prefix(usage, ruleset):
    short_ruleset = ruleset[:3] + ruleset[3:].replace('.0', '')
    return f'https://github.com/mlcommons/{usage}_results_v{short_ruleset}'


def _details_url(system_desc, usage, ruleset):
    return '{ruleset_prefix}/blob/master/{submitter}/systems/{system}.json'.format(
        ruleset_prefix=_ruleset_url_prefix(usage, ruleset),
        submitter=system_desc['submitter'],
        system=_linkable_system_name(system_desc),
    )


def _code_url(system_desc, usage, ruleset):
    return '{ruleset_prefix}/blob/master/{submitter}/benchmarks'.format(
        ruleset_prefix=_ruleset_url_prefix(usage, ruleset),
        submitter=system_desc['submitter'],
    )


def _get_sort_by_column_names():
    return [
        'division', 'system', 'accelerator_model_name', 'framework',
        'accelerators_count'
    ]


def _read_result_file(result_file, usage, ruleset):
    config_file = f'{usage}_{ruleset}/common.yaml'
    checker = mlp_compliance.make_checker(
        usage=usage,
        ruleset=ruleset,
        quiet=True,
        werror=False,
    )
    valid, _, _, _ = mlp_compliance.main(result_file, config_file, checker)
    if not valid:
        raise ValueError('Compliance check failed')

    loglines, failed = parse_file(result_file, ruleset=ruleset)
    if len(failed) > 0:
        raise ValueError('Parse error')

    return loglines


def _query_run_start_stop(loglines):
    run_start, run_stop = None, None
    for logline in loglines:
        if logline.key == 'run_start':
            run_start = logline.timestamp
        if logline.key == 'run_stop':
            run_stop = logline.timestamp
        if run_start is not None and run_stop is not None:
            break
    if run_start is None:
        raise ValueError('run_start not recorded')
    if run_stop is None:
        raise ValueError('run_stop not recorded')

    return float(run_start), float(run_stop)


def _query_mlperf_strong_scaling_score(loglines):
    run_start, run_stop = _query_run_start_stop(loglines)
    seconds = run_stop - run_start
    minutes = seconds / 60 / 1000  # convert ms to minutes
    return minutes


def _query_instance_scale(loglines):
    number_of_nodes, accelerators_per_node = None, None
    for logline in loglines:
        if logline.key == 'number_of_nodes':
            number_of_nodes = logline.value['value']
        if logline.key == 'accelerators_per_node':
            accelerators_per_node = logline.value['value']
        if number_of_nodes is not None and accelerators_per_node is not None:
            break
    if number_of_nodes is None:
        raise ValueError('number_of_nodes not recorded')
    if accelerators_per_node is None:
        raise ValueError('accelerators_per_node not recorded')
    return int(number_of_nodes) * max(int(accelerators_per_node), 1)


def _compute_olympic_average(scores, dropped_scores, max_dropped_scores):
    """Olympic average by dropping the top and bottom max_dropped_scores:
    If max_dropped_scores == 1, then we compute a normal olympic score.
    If max_dropped_scores > 1, then we drop more than one scores from the
    top and bottom and average the rest.
    When dropped_scores > 0, then some scores have already been dropped
    so we should not double count them
    Precondition: Dropped scores have higher score value than the rest
    """

    # Sort scores first
    scores.sort()

    # Remove top and bottom scores
    countable_scores = scores[max_dropped_scores:(
        len(scores) - (max_dropped_scores - dropped_scores))]
    sum_of_scores = sum(countable_scores)
    return sum_of_scores * 1.0 / len(countable_scores)


def _is_organization_folder(folder):
    if not os.path.isdir(folder):
        return False
    systems_folder = os.path.join(folder, 'systems')
    if not os.path.exists(systems_folder):
        return False
    results_folder = os.path.join(folder, 'results')
    if not os.path.exists(results_folder):
        return False
    return True


class Summary:
    def __init__(self, column_names):
        self._column_names = tuple(column_names)
        self._results = {cn: [] for cn in self._column_names}

    def push(self, column_name, value):
        assert column_name in self._column_names
        self._results[column_name].append(value)

    def to_dataframe(self):
        return pd.DataFrame(self._results, columns=self._column_names)

    def __len__(self):
        num_rows = None
        for _, values in self._results.items():
            if num_rows is None:
                num_rows = len(values)
            else:
                assert num_rows == len(values)
        return num_rows


def _get_weak_scaling_metric_schema():
    return {
        'number_of_models': float,
        'instance_scale': float,
        'time_to_train_all': float,
    }


def _get_empty_summary(usage, ruleset, weak_scaling=False):
    return Summary(
        _get_column_schema(usage, ruleset, weak_scaling=weak_scaling).keys())


def _get_column_schema(usage, ruleset, weak_scaling=False):
    schema = {
        'division': str,
        'availability': str,
        'submitter': str,
        'system': str,
        'host_processor_model_name': str,
        'host_processors_count': int,
        'accelerator_model_name': str,
        'accelerators_count': int,
        'framework': str,
        'notes': str,
    }
    if weak_scaling == True:
        benchmarks = get_allowed_benchmarks(usage, ruleset)
        for benchmark in benchmarks:
            for metric, dtype in _get_weak_scaling_metric_schema().items():
                schema['{}:{}'.format(benchmark, metric)] = dtype
    else:
        schema.update(
            {b: float
             for b in get_allowed_benchmarks(usage, ruleset)})
    schema.update({'details_url': str, 'code_url': str})
    return schema


class FieldError(ValueError):
    pass


def _assert_in_desc_and_return(desc, desc_keys, query=None):
    if not isinstance(desc_keys, (list, tuple, set)):
        desc_keys = (desc_keys, )
    if query is None:
        assert len(desc_keys) == 1
    for desc_key in desc_keys:
        if desc_key not in desc:
            raise FieldError('ERROR: "{}" field missing'.format(desc_key))
    return desc[desc_keys[0]] if query is None else query(desc)


def _get_scaling_factor(folder):
    """Apply scaling factor if benchmark dir has json file scaling.json.

    This file can be added only during the review period if the review
    committee decides score should be scaled.
    """
    scaling_factor = 1.0
    scaling_file = os.path.join(folder, 'scaling.json')
    if os.path.isfile(scaling_file):
        contents = _read_json_file(scaling_file)
        scaling_factor = contents['scaling_factor']
        print(f'NOTICE: Applying scaling factor {scaling_factor} to dir {folder}')
    return scaling_factor


def _compute_strong_scaling_scores(desc, system_folder, usage, ruleset):
    # Collect scores for benchmarks.
    benchmark_scores = {}
    benchmark_folder_parent = os.path.join(
        system_folder, 'strong') if usage == 'hpc' else system_folder
    if not os.path.isdir(benchmark_folder_parent):
        return benchmark_scores
    for benchmark_folder in _get_sub_folders(benchmark_folder_parent):
        folder_parts = benchmark_folder.split('/')
        benchmark = _benchmark_alias(folder_parts[-1])
        system = folder_parts[-3] if usage == 'hpc' else folder_parts[-2]
        # Read scores from result files.
        pattern = '{folder}/result_*.txt'.format(folder=benchmark_folder)
        result_files = glob.glob(pattern, recursive=True)
        scores = []
        dropped_scores = 0
        for result_file in result_files:
            try:
                loglines = _read_result_file(result_file, usage, ruleset)
                scores.append(_query_mlperf_strong_scaling_score(loglines))
            except ValueError as e:
                print('{} in {}'.format(e, result_file))
                dropped_scores += 1
                continue
        max_dropped_scores = 4 if benchmark == 'unet3d' else 1
        if dropped_scores > max_dropped_scores:
            print('CRITICAL ERROR: Too many non-converging runs '
                  'for {} {}/{}'.format(desc['submitter'], system, benchmark))
            print('** CRITICAL ERROR ** Results in the table for {} {}/{} are '
                  'NOT correct'.format(desc['submitter'], system, benchmark))
        elif dropped_scores >= 1:
            print('NOTICE: Dropping non-converged run(s) for {} {}/{} using '
                  'olympic scoring.'.format(
                      desc['submitter'],
                      system,
                      benchmark,
                  ))

        if dropped_scores <= max_dropped_scores:
            benchmark_scores[benchmark] = _compute_olympic_average(
                scores, dropped_scores, max_dropped_scores)

            scaling_factor = _get_scaling_factor(benchmark_folder)
            benchmark_scores[benchmark] *= scaling_factor

    _fill_empty_benchmark_scores(benchmark_scores, usage, ruleset)
    return benchmark_scores


def _compute_weak_scaling_scores(desc, system_folder, usage, ruleset):
    """ Weak scaling experiments aim to measure the "total training capacity" of
    a given system. Assume a system has T accelerators; it takes TTTa mins to
    train all M models until convergence, where each model needs S accelerators.
    Therefore, instead of a single metric TTT, each benchmark now requires a
    tuple of 4 values to be reported: (T, M, S, TTTa).

    As such, this function determines the M, S and TTTa from the result logs
    (note that T is provided by the system desc json). If a result log does not
    meet compliance, this model will not be counted.

    Note:
        T: accelerators_count
        M: number_of_models
        S: instance_scale
        TTTa: time_to_train_all
    """
    assert usage == 'hpc'
    # Collect scores for benchmarks.
    benchmark_scores = {}
    benchmark_folder_parent = os.path.join(system_folder, 'weak')
    if not os.path.isdir(benchmark_folder_parent):
        return benchmark_scores
    for benchmark_folder in _get_sub_folders(benchmark_folder_parent):
        folder_parts = benchmark_folder.split('/')
        benchmark = _benchmark_alias(folder_parts[-1])
        system = folder_parts[-3]
        # Read scores from result files.
        pattern = '{folder}/result_*.txt'.format(folder=benchmark_folder)
        result_files = glob.glob(pattern, recursive=True)
        global_start, global_stop = float('inf'), float('-inf')
        number_of_models = 0
        instance_scale = None
        for result_file in result_files:
            try:
                loglines = _read_result_file(result_file, usage, ruleset)
                start, stop = _query_run_start_stop(loglines)
                global_start = min(global_start, start)
                global_stop = max(global_stop, stop)
                number_of_models += 1
                if instance_scale == None:
                    instance_scale = _query_instance_scale(loglines)
                else:
                    assert instance_scale == _query_instance_scale(loglines)
            except ValueError as e:
                print('{} in {}'.format(e, result_file))
                continue

        if number_of_models >= get_result_file_counts(usage)[benchmark]:
            benchmark_scores['{}:{}'.format(
                benchmark,
                'time_to_train_all',
            )] = (global_stop - global_start) / 60 / 1000
            benchmark_scores['{}:{}'.format(
                benchmark,
                'number_of_models',
            )] = number_of_models
            benchmark_scores['{}:{}'.format(
                benchmark,
                'instance_scale',
            )] = instance_scale
        else:
            print('CRITICAL ERROR: Not enough converging weak scaling runs '
                  'for {} {}/{}'.format(desc['submitter'], system, benchmark))

    _fill_empty_benchmark_scores(benchmark_scores,
                                 usage,
                                 ruleset,
                                 weak_scaling=True)
    return benchmark_scores


def _load_system_desc(folder, system):
    systems_folder = os.path.join(folder, 'systems')
    system_file = os.path.join(systems_folder, '{}.json'.format(system))
    if not os.path.exists(system_file):
        raise FileNotFoundError('ERROR: Missing {}'.format(system_file))
    return _read_json_file(system_file)


def _fill_empty_benchmark_scores(
    benchmark_scores,
    usage,
    ruleset,
    weak_scaling=False,
):
    for benchmark in get_allowed_benchmarks(usage, ruleset):
        if weak_scaling:
            for metric in _get_weak_scaling_metric_schema().keys():
                k = '{}:{}'.format(benchmark, metric)
                if k not in benchmark_scores:
                    benchmark_scores[k] = None

        else:
            if benchmark not in benchmark_scores:
                benchmark_scores[benchmark] = None


def summarize_results(folder, usage, ruleset, csv_file=None):
    """Summarizes a set of results.

    Args:
        folder: The folder for a submission package.
        ruleset: The ruleset such as 0.6.0, 0.7.0, or 1.0.0.
    """
    results_folder = os.path.join(folder, 'results')

    strong_scaling_summary = _get_empty_summary(usage, ruleset)
    weak_scaling_summary = _get_empty_summary(usage,
                                              ruleset,
                                              weak_scaling=True)

    for system_folder in _get_sub_folders(results_folder):
        folder_parts = system_folder.split('/')
        system = folder_parts[-1]
        # Load corresponding system description.
        try:
            desc = _load_system_desc(folder, system)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(e)
            continue

        system_specs = {}

        def _check_and_update_system_specs(desc_keys, column_name, query=None):
            system_specs[column_name] = _assert_in_desc_and_return(
                desc,
                desc_keys,
                query=query,
            )

        # Construct prefix portion of the row.
        try:
            _check_and_update_system_specs('division', 'division')
            _check_and_update_system_specs('status', 'availability')
            _check_and_update_system_specs('submitter', 'submitter')
            _check_and_update_system_specs('system_name',
                                           'system',
                                           query=_pretty_system_name)
            _check_and_update_system_specs('host_processor_model_name',
                                           'host_processor_model_name')
            _check_and_update_system_specs(
                [
                    'host_processor_core_count', 'host_processors_per_node',
                    'number_of_nodes'
                ],
                'host_processors_count',
                query=lambda desc: (int(desc['host_processors_per_node']) *
                                    int(desc['number_of_nodes'])),
            )
            _check_and_update_system_specs(
                'accelerator_model_name',
                'accelerator_model_name',
                query=_pretty_accelerator_model_name,
            )
            _check_and_update_system_specs(
                ['accelerators_per_node', 'number_of_nodes'],
                'accelerators_count',
                query=lambda desc: int(desc['accelerators_per_node']) * int(
                    desc['number_of_nodes']),
            )
            _check_and_update_system_specs('framework',
                                           'framework',
                                           query=_pretty_framework)
            _check_and_update_system_specs(
                ['hw_notes', 'sw_notes'],
                'notes',
                query=lambda desc: ';'.join(s for s in (
                    desc['hw_notes'],
                    desc['sw_notes'],
                ) if s),
            )
        except FieldError as e:
            print("{} in {}".format(e, system_file))
            continue

        # Compute the scores.
        strong_scaling_scores = _compute_strong_scaling_scores(
            desc, system_folder, usage, ruleset)
        if usage == 'hpc':
            weak_scaling_scores = _compute_weak_scaling_scores(
                desc, system_folder, usage, ruleset)

        # Construct postfix portion of the row.
        urls = {
            'details_url': _details_url(desc, usage, ruleset),
            'code_url': _code_url(desc, usage, ruleset)
        }
        # Update the summaries.
        if len(strong_scaling_scores) > 0:
            for column_name, value in itertools.chain(
                    system_specs.items(),
                    strong_scaling_scores.items(),
                    urls.items(),
            ):
                strong_scaling_summary.push(column_name, value)
        if usage == 'hpc' and len(weak_scaling_scores) > 0:
            for column_name, value in itertools.chain(
                    system_specs.items(),
                    weak_scaling_scores.items(),
                    urls.items(),
            ):
                weak_scaling_summary.push(column_name, value)

    # Print rows in order of the sorted keys.
    strong_scaling_summary = strong_scaling_summary.to_dataframe().sort_values(
        _get_sort_by_column_names()).reset_index(drop=True)
    if len(weak_scaling_summary) > 0:
        weak_scaling_summary = weak_scaling_summary.to_dataframe().sort_values(
            _get_sort_by_column_names()).reset_index(drop=True)
    return strong_scaling_summary, weak_scaling_summary


def get_parser():
    parser = argparse.ArgumentParser(
        prog='mlperf_logging.result_summarizer',
        description='Summarize a set of result files.',
    )

    parser.add_argument('folder',
                        type=str,
                        help='the folder for a submission package')
    parser.add_argument(
        'usage',
        type=str,
        default="training",
        choices=usage_choices(),
        help='the usage such as training, hpc, inference_edge, inference_server'
    )
    parser.add_argument('ruleset',
                        type=str,
                        choices=rule_choices(),
                        help='the ruleset such as 0.6.0, 0.7.0, or 1.0.0')
    parser.add_argument('--werror',
                        action='store_true',
                        help='Treat warnings as errors')
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress warnings. Does nothing if --werror is set')
    parser.add_argument(
        '-csv',
        '--csv',
        type=str,
        help='Exports a csv of the results to the path specified')

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    strong_scaling_summaries = []
    weak_scaling_summaries = []

    def _update_summaries(folder):
        strong_scaling_summary, weak_scaling_summary = summarize_results(
            folder,
            args.usage,
            args.ruleset,
        )
        strong_scaling_summaries.append(strong_scaling_summary)
        if len(weak_scaling_summary) > 0:
            weak_scaling_summaries.append(weak_scaling_summary)

    multiple_folders_regex = r'(.*)\{(.*)\}'
    multiple_folders = re.search(multiple_folders_regex, args.folder)
    if multiple_folders:
        # Parse results for multiple organizations.
        path_prefix = multiple_folders.group(1)
        path_suffix = multiple_folders.group(2)
        if ',' in path_suffix:
            orgs = multiple_folders.group(2).split(',')
        elif '*' == path_suffix:
            orgs = os.listdir(path_prefix)
            orgs = [
                org for org in orgs
                if _is_organization_folder(os.path.join(path_prefix, org))
            ]
        print('Detected organizations: {}'.format(', '.join(orgs)))
        for org in orgs:
            org_folder = path_prefix + org
            _update_summaries(org_folder)
    else:
        # Parse results for single organization.
        _update_summaries(args.folder)

    # Print and write back results.
    def _print_and_write(summaries, weak_scaling=False, mode='w'):
        if len(summaries) > 0:
            summaries = pd.concat(summaries).astype(
                _get_column_schema(
                    args.usage,
                    args.ruleset,
                    weak_scaling=weak_scaling,
                ))
            if weak_scaling:
                print('Weak Scaling Scores:')
            else:
                print('Strong Scaling Scores:')
            print(summaries)
            if args.csv is not None:
                summaries.to_csv(args.csv, index=False, mode=mode)

    with pd.option_context('display.max_rows', None, 'display.max_columns',
                           None, 'display.max_colwidth', None):
        _print_and_write(strong_scaling_summaries)
        _print_and_write(weak_scaling_summaries, weak_scaling=True, mode='a')


if __name__ == '__main__':
    main()
