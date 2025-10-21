import sys
import logging

from . import mlp_compliance

parser = mlp_compliance.get_parser()
args = parser.parse_args()

logging.basicConfig(filename=args.log_output, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler())
formatter = logging.Formatter("%(levelname)s - %(message)s")
logging.getLogger().handlers[0].setFormatter(formatter)
logging.getLogger().handlers[1].setFormatter(formatter)

config_file = args.config or f'{args.usage}_{args.ruleset}/common.yaml'

checker = mlp_compliance.make_checker(
    args.usage,
    args.ruleset,
    args.quiet,
    args.werror,
)

valid, system_id, benchmark, result = mlp_compliance.main(args.filename, config_file, checker)

if not valid:
    logging.error('FAILED')
    print('** Logging output also at', args.log_output)
    sys.exit(1)
else:
    print('** Logging output also at', args.log_output)
    logging.info('SUCCESS')
