import sys
import logging

from . import rcp_checker

parser = rcp_checker.get_parser()
args = parser.parse_args()

logging.basicConfig(filename=args.log_output, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
formatter = logging.Formatter("%(levelname)s - %(message)s")
logging.getLogger().handlers[0].setFormatter(formatter)
logging.getLogger().handlers[1].setFormatter(formatter)

# Results summarizer makes these 3 calls to invoke RCP test
checker = rcp_checker.make_checker(args.rcp_usage, args.rcp_version, args.verbose, args.bert_train_samples)
checker._compute_rcp_stats()
# Check pruned RCPs by default. Use rcp_pass='full_rcp' for full check
test, msg = checker._check_directory(args.dir, rcp_pass=args.rcp_pass)

if test:
    logging.info('%s, RCP test PASSED', msg)
    print('** Logging output also at', args.log_output)
else:
    logging.error('%s, RCP test FAILED, consider adding --rcp_bypass in when running the package_checker.', msg)
    print('** Logging output also at', args.log_output)
    sys.exit(1)
