# MLPerf package checker

MLPerf package checker

## Usage

To check an organization's submission package for compliance:

```sh
python3 -m mlperf_logging.package_checker FOLDER USAGE RULESET
```

Currently, USAGE in ["training"] and RULESET in ["0.6.0", "0.7.0", "1.0.0", "1.1.0", "2.0.0"] are supported.

The package checker checks:
1. The number of result files for each benchmark matches the required count. If
   the actual and required counts do not match, an error is printed.
2. For every result file, the logging within the file is compliant. If there are
   any compliance errors, they are printed.
3. For every result directory whether the seed checker compliance is met
4. For every result directory whether the reference convergence point compliance (RCP) is met.
5. The system description json files in the systems directory for compliance.

## Tested software versions
Tested and confirmed working using the following software versions:

Python 3.7.7
Python 3.9.2
Python 3.9.10
