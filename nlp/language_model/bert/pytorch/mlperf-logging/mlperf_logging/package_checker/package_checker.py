'''
Runs a set of checks on an organization's submission package.
'''
from __future__ import print_function

import argparse
import glob
import json
import logging
import os
import sys

from ..compliance_checker import mlp_compliance
from ..compliance_checker.mlp_compliance import usage_choices, rule_choices
from ..rcp_checker import rcp_checker
from .seed_checker import find_source_files_under, SeedChecker
from ..system_desc_checker import system_desc_checker

from ..benchmark_meta import get_allowed_benchmarks, get_result_file_counts



def _get_sub_folders(folder):
    sub_folders = [
        os.path.join(folder, sub_folder) for sub_folder in os.listdir(folder)
    ]
    return [
        sub_folder for sub_folder in sub_folders if os.path.isdir(sub_folder)
    ]


def _print_divider_bar():
    logging.info('------------------------------')


def check_training_result_files(folder, usage, ruleset, quiet, werror,
                                rcp_bypass, rcp_bert_train_samples):
    """Checks all result files for compliance.

    Args:
        folder: The folder for a submission package.
        ruleset: The ruleset such as 0.6.0, 0.7.0, 1.0.0, etc.
    """
    allowed_benchmarks = get_allowed_benchmarks(usage, ruleset)
    benchmark_file_counts = get_result_file_counts(usage)


    seed_checker = SeedChecker(ruleset)
    too_many_errors = False
    result_folder = os.path.join(folder, 'results')
    for system_folder in _get_sub_folders(result_folder):
        if usage == 'hpc':
            benchmark_folders = []
            for scaling_folder in _get_sub_folders(system_folder):
                benchmark_folders.extend(_get_sub_folders(scaling_folder))
        else:
            benchmark_folders = _get_sub_folders(system_folder)
        for benchmark_folder in benchmark_folders:
            folder_parts = benchmark_folder.split('/')
            benchmark = folder_parts[-1]
            if usage == 'hpc':
                assert folder_parts[-2] in {'strong', 'weak'}
                is_weak_scaling = (folder_parts[-2] == 'weak')
                system = folder_parts[-3]
            else:
                is_weak_scaling = False
                system = folder_parts[-2]

            # Find whether submission is closed and only then run seed and RCP checkers
            system_desc_file = os.path.join(folder, 'systems/') + system + '.json'
            division = ''
            with open(system_desc_file, 'r') as f:
                contents = json.load(f)
                if contents['division'] == 'closed':
                    division = 'closed'

            # If it is not a recognized benchmark, skip further checks.
            if benchmark not in allowed_benchmarks:
                logging.warning(' Skipping benchmark: %s', benchmark)
                continue

            # Find all result files for this benchmark.
            pattern = '{folder}/result_*.txt'.format(folder=benchmark_folder)
            result_files = glob.glob(pattern, recursive=True)
            any_pattern = '{folder}/*'.format(folder=benchmark_folder)
            all_files = glob.glob(any_pattern, recursive=True)

            # Find all source codes for this benchmark.
            source_files = find_source_files_under(
                os.path.join(folder, 'benchmarks', benchmark))

            _print_divider_bar()
            logging.info(' Running compliance checks in dir %s', benchmark_folder)
            logging.info(' System %s', system)
            logging.info(' Benchmark %s', benchmark)
            _print_divider_bar()

            if is_weak_scaling:
                if len(result_files) < benchmark_file_counts[benchmark]:
                    logging.error('Expected at least %d runs, but detected %d runs.',
                        benchmark_file_counts[benchmark],
                        len(result_files))
                    too_many_errors = True
            else:
                # The number of result files must be an exact number.
                # Print a comprehensive message if some files in results
                # directory do not match naming convention (results_*.txt)
                if len(result_files) != benchmark_file_counts[benchmark]:
                    logging.error('Incorrect number of files in dir, or wrong file names in directory %s, '
                                  'found %d, expected %d',
                                  benchmark_folder, len(result_files), benchmark_file_counts[benchmark])
                    too_many_errors = True
            if len(all_files) > len(result_files):
                logging.warning('Detected %d total files in directory %s, but some do not conform '
                      'to naming convention, should you rename them to result_*.txt ?',len(all_files), benchmark_folder)

            if len(result_files) < len(all_files):
                logging.warning('Unknown files in result directory: %s', benchmark_folder)

            errors_found = 0
            result_files.sort()
            for result_file in result_files:
                result_basename = os.path.basename(result_file)
                result_name, _ = os.path.splitext(result_basename)
                run = result_name.split('_')[-1]

                # For each result file, run the benchmark's compliance checks.
                _print_divider_bar()
                logging.info('Run %d/%d', result_files.index(result_file) + 1, len(result_files))
                config_file = '{usage}_{ruleset}/common.yaml'.format(
                    usage=usage,
                    ruleset=ruleset,
                    benchmark=benchmark,
                )
                checker = mlp_compliance.make_checker(
                    usage=usage,
                    ruleset=ruleset,
                    quiet=quiet,
                    werror=werror,
                )
                valid, _, _, _ = mlp_compliance.main(
                    result_file,
                    config_file,
                    checker,
                )
                if not valid:
                    errors_found += 1
            if errors_found == 1 and benchmark != 'unet3d':
                logging.warning(" 1 file does not comply, accepting this under olympic scoring")
            elif errors_found > 0 and errors_found <= 4 and benchmark == 'unet3d':
                logging.warning(" %d files do not comply, accepting this under olympic scoring", errors_found)
            elif errors_found > 0:
                too_many_errors = True
                logging.error(" %d files do not comply, directory cannot be accepted", errors_found)

            # Check if each run use unique seeds.
            if ruleset in {'1.0.0', '1.1.0', '2.0.0'} and division == 'closed':
                if not seed_checker.check_seeds(result_files, source_files):
                    too_many_errors = True
                    logging.error('Seed checker failed')

            # Run RCP checker for >= 1.0.0
            if ruleset in {'1.0.0', '1.1.0', '2.0.0'} and division == 'closed' and benchmark != 'minigo':
                rcp_chk = rcp_checker.make_checker(usage, ruleset, verbose=False, bert_train_samples=rcp_bert_train_samples)
                rcp_chk._compute_rcp_stats()

                # Now go again through result files to do RCP checks
                rcp_pass, rcp_msg = rcp_chk._check_directory(benchmark_folder, rcp_pass='pruned_rcps', rcp_bypass=rcp_bypass)
                if not rcp_pass:
                    logging.error('RCP Test Failed: %s', rcp_msg)
                    too_many_errors = True
                else:
                    logging.info('RCP Test Passed: %s', rcp_msg)

            _print_divider_bar()

    _print_divider_bar()
    if too_many_errors:
        logging.info('PACKAGE CHECKER FOUND ERRORS, LOOK INTO ERROR LOG LINES AND FIX THEM.')
    else:
        logging.info('PACKAGE CHECKER FOUND NO ERRORS, SUCCESS !')


def check_systems(folder, usage, ruleset):
    """Checks the system decription files

    Args:
        folder: The folder for a submission package.
        usage: The usage such as training, inference_edge, inference_server, hpc.
        ruleset: The ruleset such as 0.6.0, 0.7.0, 1.0.0, etc.
    """
    system_folder = os.path.join(folder,'systems')
    pattern = '{folder}/*.json'.format(folder=system_folder)
    json_files = glob.glob(pattern)

    too_many_errors = False
    for json_file in json_files:
        valid, _, _, _ = system_desc_checker.check_system_desc(json_file, usage, ruleset)
        if not valid:
            too_many_errors = True

    return not too_many_errors

def check_training_package(folder, usage, ruleset, quiet, werror, rcp_bypass, rcp_bert_train_samples, log_output):
    """Checks a training package for compliance.

    Args:
        folder: The folder for a submission package.
        usage: The usage such as training or hpc
        ruleset: The ruleset such as 0.6.0, 0.7.0, 1.0.0, etc.
    """
    if ruleset in {'1.0.0', '1.1.0', '2.0.0'}:
        logging.info(' Checking System Description Files')
        if not check_systems(folder, usage, ruleset):
            logging.error('System description file checker failed')

    check_training_result_files(folder, usage, ruleset, quiet, werror, rcp_bypass, rcp_bert_train_samples)
    _print_divider_bar()
    print('\n** Detailed log output is also at', log_output)


def get_parser():
    parser = argparse.ArgumentParser(
        prog='mlperf_logging.package_checker',
        description='Lint MLPerf submission packages.',
    )

    parser.add_argument(
        'folder',
        type=str,
        help='the folder for a submission package',
    )
    parser.add_argument(
        'usage',
        type=str,
        choices=usage_choices(),
        help='the usage such as training, inference_edge, inference_server, hpc',
    )
    parser.add_argument(
        'ruleset',
        type=str,
        choices=rule_choices(),
        help='the ruleset such as 0.6.0, 0.7.0, 1.0.0, etc.'
    )
    parser.add_argument(
        '--werror',
        action='store_true',
        help='Treat warnings as errors',
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress warnings. Does nothing if --werror is set',
    )
    parser.add_argument(
        '--rcp_bypass',
        action='store_true',
        help='Bypass failed RCP checks so that submission uploads'
    )
    parser.add_argument(
        '--rcp_bert_train_samples',
        action='store_true',
        help='If set, num samples used for training '
             'bert benchmark is taken from train_samples, '
             'istead of epoch_num',
    )
    parser.add_argument(
        '--log_output',
        type=str,
        default='package_checker.log',
        help='where to store package checker output log'
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(filename=args.log_output, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    logging.getLogger().handlers[0].setFormatter(formatter)
    logging.getLogger().handlers[1].setFormatter(formatter)

    check_training_package(args.folder, args.usage, args.ruleset, args.quiet, args.werror,
                           args.rcp_bypass, args.rcp_bert_train_samples, args.log_output)


if __name__ == '__main__':
    main()
