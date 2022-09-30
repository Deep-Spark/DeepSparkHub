# MLPerf Logging

MLPerf Compliance Logging Utilities and Helper Functions

## Installation

Use one of the following ways to install.

- For development, you may download the latest version and install from local path:

  ```sh
  git clone https://github.com/mlperf/logging.git mlperf-logging
  pip install -e mlperf-logging
  ```

- Install from github at a specific commit (Replace TAG_OR_COMMIT_HASH with actual tag or commit hash):
  ```sh
  pip install "git+https://github.com/mlperf/logging.git@TAG_OR_COMMIT_HASH"
  ```

Uninstall:

```sh
pip uninstall mlperf-logging
```

## Packages

- [mllog](mlperf_logging/mllog): MLPerf logging library
- [compliance checker](mlperf_logging/compliance_checker): utility checking compliance of MLPerf log
- [system_desc_checker](mlperf_logging/system_desc_checker): utility checking compliance of system description json file
- [rcp_checker](mlperf_logging/rcp_checker): utility running convergence checks in submission directories
- [package_checker](mlperf_logging/package_checker): top-level checker for a package, it calls compliance checker, system desc checker, and rcp checker
- [result_summarizer](mlperf_logging/result_summarizer): utility that parses package and prints out result summary
- [repo_checker](mlperf_logging/repo_checker): utility that checks source code files for github compliance

## Instructions

A submission needs to pass the package checker and run the result summarizer.
For submission 2.0 (latest) you can do that with. For previous versions use the respective verify script.

```sh
./scripts/verify_for_v2.0_training.sh <submission_directory>
```

If you want to run the individual utilities/checker, please check the README files in the respective subdirectories.
