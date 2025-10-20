# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Generate pretrain MindRecord for BERT."""

import collections
import random
from argparse import ArgumentParser
import tokenization
import numpy as np
from mindspore.mindrecord import FileWriter


def parse_args():
    parser = ArgumentParser(description="Generate MindRecord for bert")
    parser.add_argument("--input_file", type=str, default="",
                        help="Input raw text file (or comma-separated list of files).")
    parser.add_argument("--output_file", type=str, default="",
                        help="Output MindRecord file (or comma-separated list of files).")
    parser.add_argument("--vocab_file", type=str, default="",
                        help="The vocabulary file that the BERT model was trained on.")
    parser.add_argument("--do_lower_case", type=bool, default=True,
                        help="Whether to lower case the input text. "
                             "Should be True for uncased models and False for cased models.")
    parser.add_argument("--do_whole_word_mask", type=bool, default=False,
                        help="Whether to use whole word masking rather than per-WordPiece masking.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of masked LM predictions per sequence.")
    parser.add_argument("--random_seed", type=int, default=12345, help="Random seed for data generation.")
    parser.add_argument("--dupe_factor", type=int, default=5,
                        help="Number of times to duplicate the input data (with different masks).")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15, help="Masked LM probability.")
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of creating sequences which are shorter than the maximum length.")
    args_opt = parser.parse_args()
    return args_opt


class TrainingInstance():
    """A single training instance (sentence pair)."""
    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join([tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_file):
    """Create TF example files from `TrainingInstance`s."""
    schema = {
        "input_ids": {"type": "int32", "shape": [-1]},
        "input_mask": {"type": "int32", "shape": [-1]},
        "segment_ids": {"type": "int32", "shape": [-1]},
        "masked_lm_positions": {"type": "int32", "shape": [-1]},
        "masked_lm_ids": {"type": "int32", "shape": [-1]},
        "masked_lm_weights": {"type": "float32", "shape": [-1]},
        "next_sentence_label": {"type": "int32", "shape": [-1]},
    }
    writer = FileWriter(output_file, overwrite=True)
    writer.add_schema(schema)

    total_written = 0
    for (_, instance) in enumerate(instances):
        all_data = []
        input_ids = tokenization.convert_tokens_to_ids(args.vocab_file, instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenization.convert_tokens_to_ids(args.vocab_file, instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        next_sentence_label = 1 if instance.is_random_next else 0
        input_ids = np.array(input_ids, dtype=np.int32)
        input_mask = np.array(input_mask, dtype=np.int32)
        segment_ids = np.array(segment_ids, dtype=np.int32)
        masked_lm_positions = np.array(masked_lm_positions, dtype=np.int32)
        masked_lm_ids = np.array(masked_lm_ids, dtype=np.int32)
        masked_lm_weights = np.array(masked_lm_weights, dtype=np.float32)
        next_sentence_label = np.array(next_sentence_label, dtype=np.int32)
        data = {'input_ids': input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
                "masked_lm_positions": masked_lm_positions,
                "masked_lm_ids": masked_lm_ids,
                "masked_lm_weights": masked_lm_weights,
                "next_sentence_label": next_sentence_label}
        all_data.append(data)
        if all_data:
            writer.write_raw_data(all_data)
            total_written += 1

    writer.commit()
    print("Wrote %d total instances", total_written)


def create_training_instances(input_file, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    """Create `TrainingInstance`s from raw text."""
    all_documents = [[]]

    # Input file format:
    # (1) One sentence per line. These should ideally be actual sentences, not
    # entire paragraphs or arbitrary spans of text. (Because we use the
    # sentence boundaries for the "next sentence prediction" task).
    # (2) Blank lines between documents. Document boundaries are needed so
    # that the "next sentence prediction" task doesn't span between documents.
    with open(input_file, "r") as reader:
        while True:
            line = tokenization.convert_to_unicode(reader.readline())
            if not line:
                break
            line = line.strip()

            # Empty lines are used as document delimiters
            if not line:
                all_documents.append([])
            tokens = tokenizer.tokenize(line)
            if tokens:
                all_documents[-1].append(tokens)


    # Remove empty documents
    all_documents = [x for x in all_documents if x]
    rng.shuffle(all_documents)

    vocab_words = list(tokenizer.vocab_dict.keys())
    instances = []
    for _ in range(dupe_factor):
        for document_index in range(len(all_documents)):
            instances.extend(create_instances_from_document(all_documents, document_index, max_seq_length,
                                                            short_seq_prob, masked_lm_prob, max_predictions_per_seq,
                                                            vocab_words, rng))

    rng.shuffle(instances)
    return instances


def create_instances_from_document(all_documents, document_index, max_seq_length, short_seq_prob,
                                   masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                (tokens_a, tokens_b, i, is_random_next) = init_tokena_and_tokenb(current_chunk, rng, target_seq_length,
                                                                                 all_documents, document_index, i)
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens, segment_ids = init_tokens_and_segment_ids(tokens_a, tokens_b)

                (tokens, masked_lm_positions, masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1
    return instances

def init_tokena_and_tokenb(current_chunk, rng, target_seq_length, all_documents, document_index, i):
    """init tokens_a and tokens_b"""
    # `a_end` is how many segments from `current_chunk` go into the `A`
    # (first) sentence.
    a_end = 1
    if len(current_chunk) >= 2:
        a_end = rng.randint(1, len(current_chunk) - 1)

    tokens_a = []
    for j in range(a_end):
        tokens_a.extend(current_chunk[j])

    tokens_b = []
    # Random next
    is_random_next = False
    if len(current_chunk) == 1 or rng.random() < 0.5:
        is_random_next = True
        target_b_length = target_seq_length - len(tokens_a)

        # This should rarely go for more than one iteration for large
        # corpora. However, just to be careful, we try to make sure that
        # the random document is not the same as the document
        # we're processing.
        for _ in range(10):
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
                break

        random_document = all_documents[random_document_index]
        random_start = rng.randint(0, len(random_document) - 1)
        for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
                break
        # We didn't actually use these segments so we "put them back" so
        # they don't go to waste.
        num_unused_segments = len(current_chunk) - a_end
        i -= num_unused_segments
    # Actual next
    else:
        is_random_next = False
        for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
    return (tokens_a, tokens_b, i, is_random_next)

def init_tokens_and_segment_ids(tokens_a, tokens_b):
    """init tokens and segment_ids."""
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)
    return tokens, segment_ids

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token in ('[CLS]', '[SEP]'):
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if (args.do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def main():
    tokenizer = tokenization.FullTokenizer(vocab_file=args.vocab_file, do_lower_case=args.do_lower_case)
    rng = random.Random(args.random_seed)
    print("before create_training_instances", flush=True)
    instances = create_training_instances(
        args.input_file, tokenizer, args.max_seq_length, args.dupe_factor,
        args.short_seq_prob, args.masked_lm_prob, args.max_predictions_per_seq,
        rng)

    print("done write_instance_to_example_files", flush=True)
    write_instance_to_example_files(instances, tokenizer, args.max_seq_length,
                                    args.max_predictions_per_seq, args.output_file)


if __name__ == "__main__":
    args = parse_args()
    main()
