import warnings
import os
import logging

from ..compliance_checker import mlp_parser

# What are source files?
SOURCE_FILE_EXT = {
    '.py', '.cc', '.cpp', '.cxx', '.c', '.h', '.hh', '.hpp', '.hxx', '.sh',
    '.sub', '.cu', '.cuh'
}


def _print_divider_bar():
    logging.info('------------------------------')


def is_source_file(path):
    """ Check if a file is considered as a "source file" by extensions.

    The extensions that are considered as "source file" are listed in
    SOURCE_FILE_EXT.

    Args:
        path: The absolute path, relative path or name to/of the file.
    """
    return os.path.splitext(path)[1].lower() in SOURCE_FILE_EXT


def find_source_files_under(path):
    """ Find all source files in all sub-directories under a directory.

    Args:
        path: The absolute or relative path to the directory under query.
    """
    source_files = []
    for root, subdirs, files in os.walk(path):
        for file_name in files:
            if is_source_file(file_name):
                source_files.append(os.path.join(root, file_name))
    return source_files


class SeedChecker:
    """ Check if the seeds fit MLPerf submission requirements.
    Current requirements are:

    1. All seeds must be logged through mllog (if choose to log seeds). Any seed
       logged via any other method will be discarded.
    2. All seeds, if choose to be logged, must be valid integers (convertible
       via int()).
    3. If any run logs at least one seed, we expect all runs to log at least
       one seed.
    4. If one run logs one seed on a certain line in a certain source file, no
       other run can log the same seed on the same line in the same file.

    Unsatisfying any of the above requirements results in check failure.

    A warning is raised for the following situations:

    1. Any run logs more than one seed.
    2. No seed is logged, however, the source files (after being converted to
       lowercase characters) contain the keyword "seed". What files are
       considered as source files are defined in SOURCE_FILE_EXT and
       is_source_file().
    """
    def __init__(self, ruleset):
        self._ruleset = ruleset

    def _get_seed_records(self, result_file):
        loglines, errors = mlp_parser.parse_file(
            result_file,
            ruleset=self._ruleset,
        )
        if len(errors) > 0:
            raise ValueError('\n'.join(
                ['Found parsing errors:'] +
                ['{}\n  ^^  {}'.format(line, error)
                 for line, error in errors] +
                ['', 'Log lines had parsing errors.']))
        return [(
            line.value['metadata']['file'],
            line.value['metadata']['lineno'],
            int(line.value['value']),
        ) for line in loglines if line.key == 'seed']

    def _assert_unique_seed_per_run(self, result_files):
        no_logged_seed = True
        error_messages = []
        seed_to_result_file = {}
        for result_file in result_files:
            try:
                seed_records = self._get_seed_records(result_file)
            except Exception as e:
                error_messages.append("Error found when querying seeds from "
                                      "{}: {}".format(result_file, e))
                continue

            if not no_logged_seed and len(seed_records) == 0:
                error_messages.append(
                    "Result file {} logs no seed. However, other "
                    "result files, including {}, already logs some "
                    "seeds.".format(result_file,
                                    list(seed_to_result_file.keys())))
            if no_logged_seed and len(seed_records) > 0:
                no_logged_seed = False
            if len(seed_records) > 1:
                warnings.warn(
                    "Result file {} logs more than one seeds {}!".format(
                        result_file, seed_records))
            for f, ln, s in seed_records:
                if (f, ln, s) in seed_to_result_file:
                    error_messages.append(
                        "Result file {} logs seed {} on {}:{}. However, "
                        "result file {} already logs the same seed on the same "
                        "line.".format(
                            result_file,
                            s,
                            f,
                            ln,
                            seed_to_result_file[(f, ln, s)],
                        ))
                else:
                    seed_to_result_file[(f, ln, s)] = result_file

        return no_logged_seed, error_messages

    def _has_seed_keyword(self, source_file):
        with open(source_file, 'r') as file_handle:
            for line in file_handle.readlines():
                if 'seed' in line.lower():
                    return True
        return False

    def check_seeds(self, result_files, source_files):
        """ Check the seeds for a specific benchmark submission.

        Args:
            result_files: An iterable contains paths to all the result files for
                this benchmark.
            source_files: An iterable contains paths to all the source files for
                this benchmark.

        """
        _print_divider_bar()
        logging.info(" Running Seed Checker")
        no_logged_seed, error_messages = self._assert_unique_seed_per_run(
            result_files)

        if len(error_messages) > 0:
            logging.error(" Seed checker failed and found the following errors: %s", '\n'.join(error_messages))
            #print("Seed checker failed and found the following "
            #      "errors:\n{}".format('\n'.join(error_messages)))
            return False

        if no_logged_seed:
            for source_file in source_files:
                if self._has_seed_keyword(source_file):
                    warnings.warn(
                        "Source file {} contains the keyword 'seed' but no "
                        "seed value is logged!".format(source_file))
        return True
