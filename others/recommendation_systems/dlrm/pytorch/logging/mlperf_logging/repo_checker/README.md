# MLPerf repository checker

MLPerf repository checker

## Usage

To check whether an organization's submission package is compatible with github
and whether it will cause any problems when added to github with a PR during the
review process.

```sh
python3 -m mlperf_logging.repo_checker FOLDER USAGE RULESET
```

Currently, USAGE in ["training"] and only RULESET 2.0.0 is supported.

The repo checker checks:
1. Whether the repo contains filenames that github does not like, e.g. files with spaces,
   files that start with '.' or '/.'
2. Files that violate the github file limit (50MB) 

## Tested software versions
Tested and confirmed working using the following software versions:

Python 3.9.10
