# Copyright (c) 2022 Iluvatar CoreX. All rights reserved.
# Copyright Declaration: This software, including all of its code and documentation,
# except for the third-party software it contains, is a copyrighted work of Shanghai Iluvatar CoreX
# Semiconductor Co., Ltd. and its affiliates ("Iluvatar CoreX") in accordance with the PRC Copyright
# Law and relevant international treaties, and all rights contained therein are enjoyed by Iluvatar
# CoreX. No user of this software shall have any right, ownership or interest in this software and
# any use of this software shall be in compliance with the terms and conditions of the End User
# License Agreement.


import json
from typing import Mapping

from dltest.log_parser import LogParser
from dltest.cli.log_parser_cli import LogParserCLI


class FetchLog(LogParserCLI):

    def command_name(self):
        return "fetch"

    def predefine_args(self):
        super(FetchLog, self).predefine_args()
        self.parser.add_argument('log', type=str, help="Log path")
        self.parser.add_argument('--saved', type=str, default=None, help='Save to path')

    def run(self):
        args = self.parse_args()
        parser = LogParser(
            patterns=args.patterns, pattern_names=args.pattern_names,
            use_re=args.use_re, nearest_distance=args.nearest_distance,
            start_line_pattern_flag=args.start_flag,
            end_line_pattern_flag=args.end_flag,
            split_pattern=args.split_pattern,
            split_sep=args.split_sep,
            split_idx=args.split_idx
        )

        results = parser.parse(args.log)
        if not isinstance(results, Mapping):
            results = dict(results=results)
        print(results)

        if args.saved is not None:
            with open(args.saved, 'w') as f:
                json.dump(results, f)

