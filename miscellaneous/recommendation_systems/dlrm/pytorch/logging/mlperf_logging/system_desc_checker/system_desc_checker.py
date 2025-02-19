'''
Runs a set of checks on a <system_desc_id>.json file.
'''

from __future__ import print_function

import argparse
import json
import logging
import sys

from ..compliance_checker.mlp_compliance import usage_choices, rule_choices

def _get_or_default(json, field, default):
    if field in json:
        return json[field]
    else:
        return default


def check_system_desc(json_file, usage, ruleset):
    """Checks a training system desc json file for validity.

    Args:
        json_file: The system desc json file to check.
        usage: The usage such as training, inference_edge, inference_server, hpc.
        ruleset: The ruleset such as 0.6.0, 0.7.0, or 1.0.0.

    Returns:
        Boolean: if the system desc json is valid.
        String: system desc id.
        String: csv string ending in "," for all fields that precede results in
            summary table.
        String: csv string starting with "," for all fields that follow results
            in summary table.
    """
    with open(json_file, 'r') as f:
        contents = json.load(f)

    valid = True
    invalid_reasons = []

    required_fields = [
        "submitter",
        "division",
        "status",
        "system_name",
        "number_of_nodes",
        "host_processors_per_node",
        "host_processor_model_name",
        "host_processor_core_count",
        "host_processor_vcpu_count",
        "host_processor_frequency",
        "host_processor_caches",
        "host_processor_interconnect",
        "host_memory_capacity",
        "host_storage_type",
        "host_storage_capacity",
        "host_networking",
        "host_networking_topology",
        "host_memory_configuration",
        "accelerators_per_node",
        "accelerator_model_name",
        "accelerator_host_interconnect",
        "accelerator_frequency",
        "accelerator_on-chip_memories",
        "accelerator_memory_configuration",
        "accelerator_memory_capacity",
        "accelerator_interconnect",
        "accelerator_interconnect_topology",
        "cooling",
        "hw_notes",
        "framework",
        "other_software_stack",
        "operating_system",
        "sw_notes",
    ]

    # Check if all required fields are contained in json.
    for field in required_fields:
        if field not in contents:
            valid = False
            invalid_reasons.append("Missing field {}".format(field))

    system_name = _get_or_default(contents, "system_name", "")

    table_csv_prefix = ",".join([
        _get_or_default(contents, "submitter", ""),
        _get_or_default(contents, "system_name", ""),
        _get_or_default(contents, "host_processor_model_name", ""),
        _get_or_default(contents, "host_processor_core_count", ""),
        _get_or_default(contents, "accelerator_model_name", ""),
        _get_or_default(contents, "accelerators_per_node", ""),
        _get_or_default(contents, "framework", ""),
    ]) + ","

    ruleset_prefix = "https://github.com/mlperf/{}_results_v{}".format(usage, ruleset)
    if "submitter" in contents and "system_name" in contents:
        details_link = "{ruleset_prefix}/blob/master/{submitter}/systems/{system_name}.json".format(
            ruleset_prefix=ruleset_prefix,
            submitter=contents["submitter"],
            system_name=contents["system_name"])
    else:
        details_link = ""
    code_link = ""
    if "submitter" in contents:
        code_link = "{ruleset_prefix}/blob/master/{submitter}/benchmarks".format(
            ruleset_prefix=ruleset_prefix,
            submitter=contents["submitter"])
    notes = ""
    table_csv_postfix = "," + ",".join([
        details_link,
        code_link,
        notes,
    ])

    if not valid:
        logging.error('  System description checker failed for %s : %s', system_name, invalid_reasons)
    else:
        logging.info('  System description checker passed for %s', system_name)

    return valid, system_name, table_csv_prefix, table_csv_postfix


def get_parser():
    parser = argparse.ArgumentParser(
        prog='mlperf_logging.system_desc_checker',
        description='Lint MLPerf system description files.',
    )

    parser.add_argument('filename', type=str,
                    help='the file to check for compliance')
    parser.add_argument('usage', type=str, choices=usage_choices(),
                    help='the usage such as training, inference_edge, inference_server, hpc')
    parser.add_argument('ruleset', type=str, choices=rule_choices(),
                    help='the ruleset such as 0.6.0, 0.7.0, or 1.0.0')
    parser.add_argument('--werror', action='store_true',
                    help='Treat warnings as errors')
    parser.add_argument('--quiet', action='store_true',
                    help='Suppress warnings. Does nothing if --werror is set')
    parser.add_argument('--log_output', type=str, default='system_desc_checker.log',
                    help='where to store system description checker output log')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(filename=args.log_output, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())
    formatter = logging.Formatter("%(levelname)s - %(message)s")
    logging.getLogger().handlers[0].setFormatter(formatter)
    logging.getLogger().handlers[1].setFormatter(formatter)

    check_system_desc(args.filename, args.usage, args.ruleset)
    print('** Logging output also at', args.log_output)


if __name__ == '__main__':
    main()
