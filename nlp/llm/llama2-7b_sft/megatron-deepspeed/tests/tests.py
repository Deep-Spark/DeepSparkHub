import copy
import dataclasses
import enum
import glob
import os
import subprocess
import sys
from argparse import ArgumentParser
from typing import List, Union, Optional

REQUIREMENTS_PY = ["tabulate"]
DEFAULT_LOG_DIR = "./test_logs"


def parse_args():
    parser = ArgumentParser("Test Application")
    parser.add_argument("--files", nargs='+', type=str,
                        help="test files or directions.")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR,
                        help="log dir")
    parser.add_argument("--timeout_per_case", type=int, default=None,
                        help="timeout for per case")
    parser.add_argument("--ignore_timeout", action="store_true",
                        help="ignore timeoue case when detect return code")
    parser.add_argument("--excludes", type=str, default=None,
                        help="excludes file or dir, using comma to split")
    parser.add_argument("--master_addr", type=str, default=None,
                        help="master node address")
    parser.add_argument("--master_port", type=str, default=None,
                        help="master node port")
    parser.add_argument("--nnodes", type=int, default=None,
                        help="total nodes")
    parser.add_argument("--node_rank", type=int, default=None,
                        help="this node`s rank in nodes")

    args = parser.parse_args()

    if args.files is None:
        raise RuntimeError(f"Got invalid files {args.files}.")

    if isinstance(args.files,str):
        args.files = args.files.splitlines()
    if isinstance(args.excludes,str):
        args.excludes = args.excludes.splitlines()


    print(args)

    return args


def current_dir():
    return os.path.abspath(os.path.join(__file__, ".."))


def setup():
    with open(os.path.join(current_dir(), "requirements.txt")) as f:
        deps = f.readlines()

    REQUIREMENTS_PY.extend(deps)

    for dep in REQUIREMENTS_PY:
        retcode = os.system(f"pip3 install {dep}")
        if retcode != 0:
            raise RuntimeError(f"Install {dep} fail.")


def get_file_name(file_path):
    if not isinstance(file_path, str):
        raise RuntimeError(f"Invalid file path {file_path}")

    return file_path.rsplit(".", maxsplit=1)[0]


def get_file_ext(file: str) -> Optional[str]:
    if "." not in file:
        return None

    return file.rsplit(".", maxsplit=1)[1]


def is_python_file(file: str):
    return file.endswith(".py")


def rename_file_ext(file: str, new_ext: str):
    if not new_ext.startswith("."):
        new_ext = f".{new_ext}"

    return f"{get_file_name(file)}{new_ext}"


def find_files(dir: str, file_pattern: str) -> List[str]:
    return glob.glob(os.path.join(dir, "**", file_pattern), recursive=True)


def find_python_test_files(dir: str) -> List[str]:
    if dir.endswith(".py"):
        return [dir]

    return find_files(dir, "test_*.py")


class LogType(enum.Enum):
    kContent = 0
    kFile = 1


@dataclasses.dataclass
class Result:
    command: str
    retcode: int
    test_file: str = None
    log: Optional[str] = None
    log_type: LogType = LogType.kFile
    exception: Optional[Exception] = None

    @property
    def success(self):
        return self.retcode == 0

    @property
    def is_timeout(self):
        return isinstance(self.exception, subprocess.TimeoutExpired)


def exec_command(command: Union[str, List], log_path, *args, **kwargs):
    if not isinstance(command, (list, tuple)):
        command = [command]
    stdout = None
    command.extend(['>', log_path, "2>&1"])
    command = " ".join(command)

    if "env" not in kwargs:
        kwargs["env"] = copy.copy(os.environ)

        kwargs["env"]["MEGATRON_TEST"] = "1"

    res = subprocess.run(command, stdout=stdout, stderr=subprocess.STDOUT, shell=True, start_new_session=True, *args, **kwargs)

    return res


def run_py_case(args, py_file, test_args: List[str] = None, log_dir: str = None, timeout=None) -> Result:
    if test_args is None:
        test_args = []

    if "test_utils.py" in py_file:
        command = f"torchrun --nproc_per_node=1 -m pytest -s {py_file} {' '.join(test_args)}"
    else:
        command = f"torchrun --nproc_per_node=8 --nnodes {args.nnodes} --node_rank {args.node_rank} \
        --master_addr {args.master_addr} --master_port {args.master_port} -m pytest -s {py_file} {' '.join(test_args)}"

    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR

    log_path = os.path.join(log_dir, rename_file_ext(os.path.basename(py_file), ".log"))

    new_log_dir = os.path.dirname(log_path)
    if not os.path.exists(new_log_dir):
        os.makedirs(new_log_dir, exist_ok=True)

    try:
        res = exec_command(command, log_path, timeout=timeout)
        result = Result(command=command, retcode=res.returncode, log=log_path, log_type=LogType.kFile)
    except Exception as ex:
        result = Result(command=command, retcode=1, log=log_path, log_type=LogType.kFile, exception=ex)

    os.system(f"cat {log_path}")

    return result


def run_py_cases(args, files, log_dir = None, timeout_per_case = None, excludes: List[str] = None) -> List[Result]:
    if log_dir is None:
        log_dir = DEFAULT_LOG_DIR

    if excludes is None:
        excludes = []

    def is_valid_test_case(file: str):

        for exc in excludes:
            if file.startswith(exc):
                return False

        return True
    files = files[0].split(' ')
    if isinstance(files, str):
        files = [files]

    if not isinstance(files, List):
        files = list(files)

    test_files = []
    for i, path in enumerate(files):
        if os.path.isfile(path) and not is_python_file(path):
            raise RuntimeError(f"Got invalid python file {path}.")

        if not os.path.isdir(path):
            test_files.append(path)
            continue

        # 处理 目录
        py_files = find_python_test_files(path)
        print(py_files)
        py_files.sort()
        test_files.extend(py_files)

    test_results = []
    for i, file in enumerate(test_files):
        print(f"Progress: {i} / {len(test_files)}, Case: {file}")

        if not is_valid_test_case(file):
            print(f"Skip {file}")
            continue

        result = run_py_case(args=args, py_file=file, log_dir=log_dir, timeout=timeout_per_case)
        result.test_file = file
        test_results.append(result)

    return test_results


def format_execption(exception: Optional[Exception]):
    if exception is None:
        return "-"

    if isinstance(exception, subprocess.TimeoutExpired):
        return f"timed out after {round(exception.timeout, 2)} seconds"

    return str(exception)


def summary(results: List[Result]):
    from tabulate import tabulate

    header = ["Index", "file", "log path", "exception"]
    success_cases = []
    failed_cases = []
    for i, result in enumerate(results):
        if result.success:
            success_cases.append([i, result.test_file, result.log, "-"])
        else:
            failed_cases.append(
                [i, result.test_file, result.log, format_execption(result.exception)]
            )

    if len(success_cases) > 0:
        print("=" * 80)
        print("= Success Cases ")
        print("=" * 80)
        print(tabulate(success_cases, headers=header, tablefmt="simple"))

    if len(failed_cases) > 0:
        print("=" * 80)
        print("= Failed Cases ")
        print("=" * 80)
        print(tabulate(failed_cases, headers=header, tablefmt="simple"))


def check_status(results: List[Result], ignore_timeout: bool):
    for result in results:
        if ignore_timeout and result.is_timeout:
            continue
        # print(result)
        if not result.success:
            exit(1)

    print("-" * 80)
    print("Pass")


if __name__ == '__main__':
    setup()

    args = parse_args()
    results = run_py_cases(args,
        args.files,
        log_dir=args.log_dir,
        excludes=args.excludes,
        timeout_per_case=args.timeout_per_case
    )
    summary(results)
    check_status(results, args.ignore_timeout)

