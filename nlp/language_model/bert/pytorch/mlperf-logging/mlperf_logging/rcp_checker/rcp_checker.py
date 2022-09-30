'''
RCP checker: Verifies convergence points of submissions by comparing them against RCPs (Reference Convergence Points)
'''

import argparse
import glob
import json
import logging
import os
import numpy as np
import re
import scipy.stats

# Number of submission runs for each benchmark
# References need 2x of these runs
# We use olympic scoring for statistics, so we reject
# both the top and bottom reference and submission numbers
submission_runs = {
    "training": {
        'bert': 10,
        'dlrm': 5,
        'maskrcnn' : 5,
        'resnet' : 5,
        'ssd' : 5,
        'unet3d' : 40,
        'rnnt': 10,
    },
    "hpc": {
        'cosmoflow': 10,
        'deepcam': 5,
        'oc20': 5,
    }
}

TOKEN = ':::MLLOG '

def _print_divider_bar():
    logging.info('------------------------------')

def _detect_eval_error(file_contents):
    for line in file_contents:
        if TOKEN not in line:
            continue
        line = re.sub(".*"+TOKEN, TOKEN, line).strip()
        if line.startswith(TOKEN):
            s = line[len(TOKEN):]
            if 'eval_error' in s:
                return True
    return False


def get_submission_epochs(result_files, benchmark, bert_train_samples):
    '''
    Extract convergence epochs (or train_samples for BERT)
    from a list of submission files
    Returns the batch size and the list of epochs (or samples) to converge
    -1 means run did not converge. Return None if > 1 files
    fail to converge
    '''
    not_converged = 0
    subm_epochs = []
    bs = -1
    use_train_samples = benchmark == 'bert' and bert_train_samples
    for result_file in result_files:
        with open(result_file, 'r', encoding='latin-1') as f:
            file_contents = f.readlines()
            use_eval_error = _detect_eval_error(file_contents)
            for line in file_contents:
                if TOKEN not in line:
                    continue
                line = re.sub(".*"+TOKEN, TOKEN, line).strip()
                if line.startswith(TOKEN):
                    str = line[len(TOKEN):]
                    if "global_batch_size" in str:
                        # Do we need to make sure global_batch_size is the same
                        # in all files? If so, this is obviously a bad submission
                        bs = json.loads(str)["value"]
                    if not use_train_samples and (((not use_eval_error) and "eval_accuracy" in str) or
                                                  (use_eval_error and "eval_error" in str)):
                        eval_accuracy_str = str
                        conv_epoch = json.loads(eval_accuracy_str)["metadata"]["epoch_num"]
                        conv_epoch = round(conv_epoch, 3)
                    if use_train_samples and "train_samples" in str:
                        eval_accuracy_str = str
                        conv_epoch = json.loads(eval_accuracy_str)["value"]
                    if "run_stop" in str:
                        # Epochs to converge is the the last epochs value on
                        # eval_accuracy line before run_stop
                        conv_result = json.loads(str)["metadata"]["status"]
                        if conv_result == "success":
                            subm_epochs.append(conv_epoch)
                        else:
                            subm_epochs.append(1e9)
                            not_converged = not_converged + 1
    if (not_converged > 1 and benchmark != 'unet3d') or (not_converged > 4 and benchmark == 'unet3d'):
        subm_epochs = None
    return bs, subm_epochs


class RCP_Checker:

    def __init__(self, usage, ruleset, verbose, bert_train_samples):
        if ruleset not in {'1.0.0', "1.1.0", "2.0.0"}:
            raise Exception('RCP Checker only supported in 1.0.0, 1.1.0, 2.0.0')
        self.usage = usage
        self.ruleset = ruleset
        self.alpha = 0.05
        self.tolerance = 0.0001
        self.verbose = verbose
        self.rcp_data = {}
        self.bert_train_samples = bert_train_samples
        self.submission_runs = submission_runs[usage]

        for benchmark in self.submission_runs.keys():
            raw_rcp_data = self._consume_json_file(usage, ruleset, benchmark)
            processed_rcp_data = self._process_raw_rcp_data(raw_rcp_data)
            sorted_rcp_data = dict(sorted(processed_rcp_data.items(), key = lambda item: item[1]['BS']))
            self.rcp_data.update(sorted_rcp_data)


    def _consume_json_file(self, usage, ruleset, benchmark):
        '''Read json file'''
        json_file = os.path.join(os.path.dirname(__file__),
                                 f"{usage}_{ruleset}",
                                 f"rcps_{benchmark}.json"
        )
        #json_file = os.getcwd() + '/mlperf_logging/rcp_checker/' + ruleset + '/rcps_'+ benchmark+ '.json'
        with open(json_file, 'r') as f:
            return json.load(f)


    def _process_raw_rcp_data(self, raw_rcp_data):
        '''
        Load the raw json file data into a dictionary
        that also contains mean, stdev, and max speedup for each record
        '''
        processed_rcps = {}
        for record, record_contents in raw_rcp_data.items():
            processed_record = {'Benchmark': record_contents['Benchmark'],
                                'BS': record_contents['BS'],
                                'Hyperparams': record_contents['Hyperparams'],
                                'Epochs to converge': record_contents['Epochs to converge'],
                                'RCP Mean': 0.0,
                                'RCP Stdev': 0.0,
                                'Max Speedup': 0.0}
            processed_rcps[record] = processed_record
            # TBD: Sanity check RCPs, eg number of runs, duplicate RCPs, RCPs or unknown benchmark
            # numbers out of bounds, etc.
        return processed_rcps


    def _prune_rcps(self):
        '''
        Prune RCPs. We compare convergence of each RCP point with interpolation using surrounding points
        and move RCP points that have min (fastest) convergence to pruned_rcp_data.
        pruned_rcp_data is by default used for RCP tests.
        '''
        self.pruned_rcp_data = {}
        for benchmark in submission_runs['training'].keys():
            min_epochs = []
            for record, record_contents in self.rcp_data.items():
                if record_contents['Benchmark'] == benchmark:
                    min_epochs.append(record_contents)

            # Step 1
            # Find point with fastest convergence and prune all point with smaller batch size
            # In that way the min batch size point will have the fastest convergenece
            fastest_conv = min(min_epochs, key=lambda rc: rc['Min Epochs'])
            min_epochs = list(filter(lambda rc: rc['BS'] >= fastest_conv['BS'], min_epochs))

            # Step 2
            # Run this algorithm for the rest of the points:
            # for i = 1..N-2
            #    if RCP[i+1] has slower convergence than interpolation(RCP[i], RCP[i+2]):
            #      remove it
            #      decrement i,N
            list_len = len(min_epochs)
            i = 1
            while i < list_len - 1:
                rcp_min = min_epochs[i-1]
                rcp_max = min_epochs[i+1]
                bs = min_epochs[i]['BS']
                name, rcp = self._create_interp_rcp(benchmark,bs,rcp_min,rcp_max)
                if min_epochs[i]['Min Epochs'] > rcp['Min Epochs']:
                    del min_epochs[i]
                    i = i-1
                    list_len = list_len - 1
                i = i+1

            for min_epoch in min_epochs:
                for record, record_contents in self.rcp_data.items():
                    if record_contents['Benchmark'] == min_epoch['Benchmark'] and record_contents['BS'] == min_epoch['BS']:
                        self.pruned_rcp_data[record] = record_contents


    def _compute_rcp_stats(self):
        '''Compute RCP mean, stdev and min acceptable epochs for RCPs'''
        for record, record_contents in self.rcp_data.items():
            epoch_list = record_contents['Epochs to converge']
            # Use olympic mean
            epoch_list.sort()
            samples_rejected = 4 if record_contents['Benchmark'] == 'unet3d' else 1
            record_contents['RCP Mean'] = np.mean(epoch_list[samples_rejected:len(epoch_list)-samples_rejected])
            record_contents['RCP Stdev'] = np.std(epoch_list[samples_rejected:len(epoch_list)-samples_rejected])
            min_epochs = self._find_min_acceptable_mean(
                              record_contents['Benchmark'],
                              record_contents['RCP Mean'],
                              record_contents['RCP Stdev'],
                              len(epoch_list)-samples_rejected*2)
            record_contents['Max Speedup'] = record_contents['RCP Mean'] / min_epochs
            record_contents['Min Epochs'] = min_epochs

            if self.verbose:
                print(record, record_contents, "\n")

        self._prune_rcps()


    def _get_rcp_data(self, rcp_pass = 'pruned_rcps'):
        if rcp_pass == 'pruned_rcps':
            rcp_data = self.pruned_rcp_data
        elif rcp_pass == 'full_rcps':
            rcp_data =  self.rcp_data
        return rcp_data


    def _find_rcp(self, benchmark, bs, rcp_pass = 'full_rcp'):
        '''Find RCP based on benchmark and batch size'''
        rcp_data = self._get_rcp_data(rcp_pass)
        for _, record_contents in rcp_data.items():
            if record_contents['Benchmark'] == benchmark and record_contents['BS'] == bs:
                return record_contents


    def _find_min_rcp(self, benchmark, rcp_pass = 'full_rcp'):
        '''Find RCP with the smallest batch size for a benchmark'''
        min_bs = 1e9
        min_record = None
        rcp_data = self._get_rcp_data(rcp_pass)
        for _, record_contents in rcp_data.items():
            if record_contents['BS'] < min_bs and record_contents['Benchmark'] == benchmark:
                min_record = record_contents
                min_bs = record_contents['BS']
        return min_record

    def _find_top_min_rcp(self, benchmark, bs, rcp_pass = 'full_rcp'):
        '''
        Find top RCP to serve as min in interpolation.
        For example, if bs = 100 and reference has bs = 10, 20, 110, 120
        this will return the RCP with bs = 20.
        '''
        min_bs = 0
        min_record = None
        rcp_data = self._get_rcp_data(rcp_pass)
        for _, record_contents in rcp_data.items():
            if record_contents['Benchmark'] == benchmark:
                if record_contents['BS'] < bs and record_contents['BS'] > min_bs:
                    min_bs = record_contents['BS']
                    min_record = record_contents
        return min_record


    def _find_bottom_max_rcp(self, benchmark, bs, rcp_pass = 'full_rcp'):
        '''
        Find bottom RCP to serve as max in interpolation.
        For example, if bs = 100 and reference has bs = 10, 20, 110, 120
        this will return the RCP with bs = 110.
        '''
        max_bs = 1e9
        max_record = None
        rcp_data = self._get_rcp_data(rcp_pass)
        for _, record_contents in rcp_data.items():
            if record_contents['Benchmark'] == benchmark:
                if record_contents['BS'] > bs and record_contents['BS'] < max_bs:
                    max_bs = record_contents['BS']
                    max_record = record_contents
        return max_record


    def _find_p_value(self, subm_mean, subm_stdev, subm_num_samples,
                      ref_mean, ref_stdev, ref_num_samples,
                      alpha=0.05):
        '''
        Do t-test between submission and reference and return p-value and
        whether it is larger than the limit
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind_from_stats.html
        '''
        t_stat, p_value = scipy.stats.ttest_ind_from_stats(
                              subm_mean, subm_stdev, subm_num_samples,
                              ref_mean, ref_stdev, ref_num_samples,
                              equal_var=True)

        # convert from 2-sided test to 1-sided test
        if t_stat < 0:
            p_value = p_value / 2
        else:
            p_value = 1 - (p_value / 2)
        return p_value > alpha, p_value


    def _find_min_acceptable_mean(self, benchmark, mean, stdev, num_samples_ref):
        '''
        Do a binary search to find the min acceptable epoch mean to converge
        The limits are 0 and the reference mean, anything above reference is acceptable
        '''

        if stdev == 0:
            return mean
        num_samples_subm = self.submission_runs[benchmark] - 2
        mean_max = mean
        mean_min = 0.0
        mean_mid = (mean_min + mean_max) / 2
        while mean_max - mean_min > self.tolerance:
            # We assume similar stdev between submission and reference
            # Samples and means are different for p-value function
            _, p_value = self._find_p_value(
                             mean_mid, stdev, num_samples_ref,
                             mean, stdev, num_samples_subm,
                             self.alpha)
            if p_value > self.alpha:
                mean_max = mean_mid
            else:
                mean_min = mean_mid
            mean_mid = (mean_min + mean_max) / 2

        return mean_mid


    def _create_interp_rcp(self, benchmark, target_bs, low_rcp, high_rcp):
        '''
        Create an interpolation RCP for batch size target_bs by interpolating
        low_rcp and high_rcp. Add the RCP into rcp_data.
        This RCP is marked as _interp_ in its name so it does not have epochs or hparams
        '''
        mean = np.interp(
                   target_bs,
                   [low_rcp['BS'], high_rcp['BS']],
                   [low_rcp['RCP Mean'], high_rcp['RCP Mean']])
        stdev = np.interp(
                    target_bs,
                    [low_rcp['BS'], high_rcp['BS']],
                    [low_rcp['RCP Stdev'], high_rcp['RCP Stdev']])

        min_epochs = self._find_min_acceptable_mean(
                         benchmark,
                         mean,
                         stdev,
                         self.submission_runs[benchmark]*2)
        interp_record_name = benchmark + '_interp_' + str(target_bs)
        interp_record = {'Benchmark': benchmark,
                         'BS': target_bs,
                         'Hyperparams': {},
                         'Epochs to converge': [],
                         'RCP Mean': mean,
                         'RCP Stdev': stdev,
                         'Max Speedup': mean / min_epochs,
                         'Min Epochs': min_epochs}
        if self.verbose:
            logging.info(" Creating interpolation record")
            logging.info(" Low RCP: %s", low_rcp)
            logging.info(" High RCP: %s", high_rcp)
            logging.info(" Intepolation record: %s", interp_record)
        return interp_record_name, interp_record


    def _eval_submission_record(self, rcp_record, subm_epochs):
        '''Compare reference and submission convergence.'''
        subm_epochs.sort()
        samples_rejected = 4 if rcp_record["Benchmark"] == 'unet3d' else 1
        mean_subm_epochs = np.mean(subm_epochs[samples_rejected:len(subm_epochs)-samples_rejected])
        if mean_subm_epochs >= (rcp_record["RCP Mean"] / rcp_record["Max Speedup"]):
            logging.info(" RCP Record: %s", rcp_record)
            logging.info(" Submission mean epochs: %.4f", mean_subm_epochs)
            return(True)
        else:
            logging.info(" RCP Record: %s", rcp_record)
            logging.info(" Submission mean epochs: %.4f", mean_subm_epochs)
            return(False)


    def _check_directory(self, dir, rcp_pass='full_rcp', rcp_bypass=False):
        '''
        Check directory for RCP compliance.
        Returns (Pass/Fail, string with explanation)
        Possible cases, the top 3 fail before RCP check.
        - (False) Fail / did not find global_batch_size in log
        - (False) Fail / run failed to converge
        - (False) Fail / Benchmark w/o RCP records
        - (True) Pass / RCP found
        - (True) Pass / RCP interpolated
        - (True) Pass / RCP missing but submission converges slower on smaller batch size
        - (False --> True with --rcp_bypass when running from package checker) Fail / RCP found
        - (False --> True with --rcp_bypass when running from package checker) Fail / RCP interpolated
        - (False --> True with --rcp_bypass when running from package checker) Missing RCP / Submit missing RCP
        '''
        _print_divider_bar()
        logging.info(" Running RCP Checker, pass: %s", rcp_pass)
        _print_divider_bar()
        dir = dir.rstrip("/")
        pattern = '{folder}/result_*.txt'.format(folder=dir)
        benchmark = os.path.split(dir)[1]
        result_files = glob.glob(pattern, recursive=True)
        bs, subm_epochs = get_submission_epochs(result_files, benchmark, self.bert_train_samples)

        if bs == -1:
            return False, 'Could not detect global_batch_size'
        if subm_epochs is None:
            return False, 'Insufficient convergence'

        rcp_record = self._find_rcp(benchmark, bs, rcp_pass)
        rcp_msg = ''
        if rcp_record is not None:
            rcp_msg = 'RCP found'
            rcp_check = self._eval_submission_record(rcp_record, subm_epochs)
        else:
            rcp_min = self._find_top_min_rcp(benchmark, bs, rcp_pass)
            rcp_max = self._find_bottom_max_rcp(benchmark, bs, rcp_pass)
            if rcp_min is not None and rcp_max is not None:
                rcp_msg = 'RCP Interpolation'
                interp_record_name, interp_record = self._create_interp_rcp(benchmark, bs, rcp_min, rcp_max)
                rcp_check = self._eval_submission_record(interp_record, subm_epochs)
            elif rcp_min is not None and rcp_max is None:
                rcp_msg = 'Missing RCP, please submit RCP with BS = {b}'.format(b=bs)
                rcp_check = False
            elif rcp_min is None and rcp_max is not None:
                rcp_min_record = self._find_min_rcp(benchmark, rcp_pass)
                rcp_check = self._eval_submission_record(rcp_min_record, subm_epochs)
                mean_subm_epochs = np.mean(subm_epochs[1:len(subm_epochs)-1])
                if rcp_check == False:
                   rcp_msg = 'Missing RCP, please submit RCP with BS = {b}'.format(b=bs)
                else:
                   rcp_msg = 'RCP not found but slower convergence on smaller batch size'
            else:
                rcp_check = False
                rcp_msg = 'Cannot find any RCPs'

        if rcp_bypass and not rcp_check:
            if rcp_msg == 'RCP found' or rcp_msg == 'RCP Interpolation' or rcp_msg == 'Missing RCP, please submit RCP with BS = {b}'.format(b=bs):
                rcp_msg  = rcp_msg + ' passed using rcp_bypass'
                logging.warning(' RCP test failed but allowed to proceed with RCP bypass.')
                logging.warning(' Please be ready to have this reviewed by the submission committee.')
                rcp_check = True

        return rcp_check, rcp_msg


def get_parser():
    parser = argparse.ArgumentParser(
        prog='mlperf_logging.rcp_checker',
        description='Run RCP Checks on logs.',
    )

    parser.add_argument('dir', type=str,
                    help='the directory to check for compliance')
    parser.add_argument('--rcp_usage', type=str, default='training',
                    choices=['training', 'hpc'],
                    help='what WG does the benchmark come from to check the log against')
    parser.add_argument('--rcp_version', type=str, default='2.0.0',
                    help='what version of rules to check the log against')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--bert_train_samples', action='store_true',
                    help='If set, num samples used for training bert benchmark'
                         'is taken from train_samples, instead of epoch_num')
    parser.add_argument('--log_output', type=str, default='rcp_checker.log',
                    help='where to store RCP checker output log')
    parser.add_argument('--rcp_pass', type=str, default='pruned_rcps',
                    help='use "pruned_rcps" or "full_rcps" for convergence checks')
    return parser


def make_checker(usage, ruleset, verbose=False, bert_train_samples=False):
  return RCP_Checker(usage, ruleset, verbose, bert_train_samples)


def main(checker, dir):
    return checker._check_directory(dir, rcp_pass='pruned_rcps')
