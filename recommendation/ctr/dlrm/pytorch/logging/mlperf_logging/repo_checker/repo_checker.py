import argparse
import logging
import os
import subprocess


def _check_bad_filenames(submission_dir):
    """Checks for filename errors.
    Git does not like filenames with spaces or that start with ., or /. .
    """
    logging.info('Running git-unfriendly file name checks.')
    names = [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(submission_dir)
        for filename in filenames
        if filename.startswith(".") or "/." in filename or " " in filename
    ]
    if len(names) > 0:
        error = "\n".join(names)
        logging.error('Files with git-unfriendly name: %s ', error)
        logging.error('Please remove spaces from filenamed and make sure they do not start with ".", or "/."')
        return False
    return True


def _check_file_sizes(submission_dir):
    """Checks for large file sizes.
    Git does not like file sizes > 50MB.
    """
    logging.info('Running large file checks.')
    out = subprocess.run(
        [
            "find",
            submission_dir,
            "-type",
            "f",
            "-size",
            "+50M",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if len(out.stdout) != 0:
        logging.error('Files > 50MB: %s', out.stdout)
        logging.error('Please remove or reduce the size of these files.')
        return False
    return True


def run_checks(submission_dir):
    """Top-level checker function.
    Call individual checkers from this function.
    """
    logging.info('Running repository checks.')

    bad_filename_error = _check_bad_filenames(submission_dir)
    large_file_error = _check_file_sizes(submission_dir)

    if not (bad_filename_error and large_file_error):
        logging.info('CHECKS FAILED.')
        return False

    logging.info('ALL CHECKS PASSED.')
    return False


def get_parser():
    """Parse commandline."""
    parser = argparse.ArgumentParser(
        prog='mlperf_logging.repo_checker',
        description='Sanity checks to make sure that package is github compliant.',
    )

    parser.add_argument(
        'folder',
        type=str,
        help='the folder for a submission package.',
    )
    parser.add_argument(
        'usage',
        type=str,
        choices=['training'],
        help='the usage -- only training is currently supported.',
    )
    parser.add_argument(
        'ruleset',
        type=str,
        choices=['2.0.0'],
        help='the ruleset. Only 2.0.0 is currently supported.'
    )
    parser.add_argument(
        '--log_output',
        type=str,
        default='repo_checker.log',
        help='the ruleset. Only 2.0.0 is currently supported.'
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

    valid = run_checks(args.folder)
    return valid

if __name__ == '__main__':
    main()
