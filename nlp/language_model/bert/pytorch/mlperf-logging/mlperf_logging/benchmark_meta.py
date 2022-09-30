# benchmark dictionary
_ALL_RESULT_FILE_COUNTS = {
    'training': {
        'bert': 10,
        'dlrm': 5,
        'gnmt': 10,
        'maskrcnn': 5,
        'minigo': 10,
        'resnet': 5,
        'ssd': 5,
        'transformer': 10,
        'ncf': 10,
        'rnnt': 10,
        'unet3d': 40,
    },
    
    'hpc' : {
        'deepcam': 5,
        'cosmoflow': 10,
        'oc20': 5
    }
}


_ALL_ALLOWED_BENCHMARKS = {
    'training': {
        '0.6': [
            'resnet',
            'ssd',
            'maskrcnn',
            'gnmt',
            'transformer',
            'ncf',
            'minigo',
        ],
        
    '0.7': [
        'bert',
        'dlrm',
        'gnmt',
        'maskrcnn',
        'minigo',
        'resnet',
        'ssd',
        'transformer'
    ],
    '1.0': [
        'bert',
        'dlrm',
        'maskrcnn',
        'minigo',
        'resnet',
        'ssd',
        'rnnt',
        'unet3d',
    ],
    '1.1': [
        'bert',
        'dlrm',
        'maskrcnn',
        'minigo',
        'resnet',
        'ssd',
        'rnnt',
        'unet3d',
    ],
    '2.0': [
        'bert',
        'dlrm',
        'maskrcnn',
        'minigo',
        'resnet',
        'ssd',
        'rnnt',
        'unet3d',
    ],
    },
    
    'hpc': {
        '0.7': [
            'cosmoflow',
            'deepcam',
        ],
        
        '1.0': [
            'cosmoflow',
            'deepcam',
            'oc20',
        ],
    }
}


def get_allowed_benchmarks(usage, ruleset):
    # check usage
    if usage not in _ALL_ALLOWED_BENCHMARKS:
        raise ValueError('usage {} not supported!'.format(usage))

    # check ruleset
    if ruleset not in _ALL_ALLOWED_BENCHMARKS[usage]:
        # try short version:
        ruleset_short = ".".join(ruleset.split(".")[:-1])
        if ruleset_short not in _ALL_ALLOWED_BENCHMARKS[usage]:
            raise ValueError('ruleset {} is not supported in {}'.format(ruleset, usage))
        allowed_benchmarks = _ALL_ALLOWED_BENCHMARKS[usage][ruleset_short]
    else:
        allowed_benchmarks = _ALL_ALLOWED_BENCHMARKS[usage][ruleset]

    return allowed_benchmarks


def get_result_file_counts(usage):
    if usage not in _ALL_RESULT_FILE_COUNTS:
        raise ValueError('usage {} not supported!'.format(usage))
    return _ALL_RESULT_FILE_COUNTS[usage]
