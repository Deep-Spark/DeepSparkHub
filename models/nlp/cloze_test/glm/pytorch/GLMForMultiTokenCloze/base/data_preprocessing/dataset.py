# coding=utf-8

import random
import json
import pickle
from typing import Dict, List, Optional,Tuple,Union
import os
from collections import Counter
import copy
import numpy as np

TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"
TRUE_DEV_SET = "true_dev"
UNLABELED_SET = "unlabeled"

FilledPattern = Tuple[List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]


def punctuation_standardization(string: str):
    punctuation_dict = {"\u201c": "\"", "\u201d": "\"", "\u2019": "'", "\u2018": "'", "\u2013": "-"}
    for key, value in punctuation_dict.items():
        string = string.replace(key, value)
    return string

def print_rank_0(message):
    print(message, flush=True)

class InputExample(object):
    """A raw input example consisting of one or two segments of text and a label"""

    def __init__(self, guid, text_a, text_b=None, label=None, logits=None, meta: Optional[Dict] = None, idx=-1,
                 num_choices=1):
        """
        Create a new InputExample.

        :param guid: a unique textual identifier
        :param text_a: the sequence of text
        :param text_b: an optional, second sequence of text
        :param label: an optional label
        :param logits: an optional list of per-class logits
        :param meta: an optional dictionary to store arbitrary meta information
        :param idx: an optional numeric index
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.logits = logits
        self.idx = idx
        self.num_choices = num_choices
        self.meta = meta if meta else {}

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)


class RecordDateset:
    def __init__(self, args, split, tokenizer, for_train=False):
        self.processor = RecordProcessor(args)
        print_rank_0(
            f"Creating dataset from file at {args.raw_data_dir} (split={split})"
        )
        self.seq_length = args.max_seq_length
        self.tokenizer = tokenizer
        self.args = args

        if split == DEV_SET:
            example_list = self.processor.get_dev_examples(
                args.raw_data_dir, for_train=for_train)
        elif split == TRAIN_SET:
            example_list = self.processor.get_train_examples(args.raw_data_dir)
        elif split == TEST_SET:
            example_list = self.processor.get_test_examples(args.raw_data_dir)
        else:
            raise ValueError(
                f"'split' must be one of {SPLIT_TYPES}, got '{split}' instead")

        if split == TEST_SET:
            self.labeled = False
        else:
            self.labeled = True

        label_distribution = Counter(example.label for example in example_list)
        print_rank_0(
            f"Returning {len(example_list)} {split} examples with label dist.: {list(label_distribution.items())}")

        example_list.sort(key=lambda x: x.num_choices)
        self.example_list = example_list
        self.pvp = ReCordPVP(args,tokenizer, self.processor.get_labels())

    def __len__(self):
        return len(self.example_list)

    def __getitem__(self, idx):
        sample_idx = idx % len(self.example_list)
        example = self.example_list[sample_idx]
        sample = self.pvp.encode(example)
        keys = ['text', 'answer_idx', 'position', 'mask', 'target', 'logit_mask']
        sample = {key:sample[key] for key in keys}
        return sample


class RecordProcessor:
    def __init__(self, args):
        self.args = args
        self.num_truncated = 0

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.jsonl"), "train")

    def get_dev_examples(self, data_dir, for_train=False):
        return self._create_examples(os.path.join(data_dir, "val.jsonl"), "dev")

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "test.jsonl"), "test")
    
    def get_labels(self):
        return ["0", "1"]

    @staticmethod
    def _create_examples(path, set_type, seed=42, max_train_candidates_per_question: int = 10, for_train=False) -> List[
            InputExample]:
        examples = []

        entity_shuffler = random.Random(seed)

        with open(path, encoding='utf8') as f:
            for idx, line in enumerate(f):
                example_json = json.loads(line)

                idx = example_json['idx']
                text = punctuation_standardization(
                    example_json['passage']['text'])
                entities = set()

                for entity_json in example_json['passage']['entities']:
                    start = entity_json['start']
                    end = entity_json['end']
                    entity = punctuation_standardization(text[start:end + 1])
                    entities.add(entity)

                entities = list(entities)
                entities.sort()

                # we follow the GPT-3 paper wrt @highlight annotations
                text = text.replace("@highlight\n", "- ")
                questions = example_json['qas']

                for question_json in questions:
                    question = punctuation_standardization(
                        question_json['query'])
                    question_idx = question_json['idx']
                    answers = set()

                    for answer_json in question_json.get('answers', []):
                        answer = punctuation_standardization(
                            answer_json['text'])
                        answers.add(answer)

                    answers = sorted(list(answers))

                    if set_type == 'train' or for_train:
                        # create a single example per *correct* answer
                        for answer_idx, answer in enumerate(answers):
                            candidates = [
                                ent for ent in entities if ent not in answers]
                            if len(candidates) > max_train_candidates_per_question - 1:
                                entity_shuffler.shuffle(candidates)
                                candidates = candidates[:max_train_candidates_per_question - 1]

                            guid = f'{set_type}-p{idx}-q{question_idx}-a{answer_idx}'
                            meta = {
                                'passage_idx': idx,
                                'question_idx': question_idx,
                                'candidates': [answer] + candidates,
                                'answers': [answer]
                            }
                            ex_idx = [idx, question_idx, answer_idx]
                            example = InputExample(guid=guid, text_a=text, text_b=question, label="0", meta=meta,
                                                   idx=ex_idx, num_choices=len(candidates) + 1,answer_idx=[0])
                            examples.append(example)

                    else:
                        # create just one example with *all* correct answers and *all* answer candidates
                        guid = f'{set_type}-p{idx}-q{question_idx}'
                        meta = {
                            'passage_idx': idx,
                            'question_idx': question_idx,
                            'candidates': entities,
                            'answers': answers
                        }
                        example = InputExample(guid=guid, text_a=text, text_b=question, label="1", meta=meta,
                                               idx=question_idx, num_choices=len(entities),answer_idx=[entities.index(i) for i in answers])
                        examples.append(example)

        question_indices = list(
            set(example.meta['question_idx'] for example in examples))
        label_distribution = Counter(example.label for example in examples)
        print_rank_0(
            f"Returning {len(examples)} examples corresponding to {len(question_indices)} questions with label "
            f"distribution {list(label_distribution.items())}")
        return examples


class InputExample(object):
    """A raw input example consisting of one or two segments of text and a label"""

    def __init__(self, guid, text_a, text_b=None, label=None, logits=None, meta: Optional[Dict] = None, idx=-1,
                 num_choices=1,answer_idx=[]):
        """
        Create a new InputExample.

        :param guid: a unique textual identifier
        :param text_a: the sequence of text
        :param text_b: an optional, second sequence of text
        :param label: an optional label
        :param logits: an optional list of per-class logits
        :param meta: an optional dictionary to store arbitrary meta information
        :param idx: an optional numeric index
        :param answer_idx: an optional numeric index
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.logits = logits
        self.idx = idx
        self.num_choices = num_choices
        self.meta = meta if meta else {}
        self.answer_idx = answer_idx

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)


class ReCordPVP:
    def __init__(self, args, tokenizer, label_list):
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.num_truncated = 0
        self.max_seq_length = args.max_seq_length

    def encode(self, example):
        tokenizer = self.tokenizer
        raw_parts_a, raw_parts_b = self.get_parts(example)

        raw_parts_a = [x if isinstance(x, tuple) else (
            x, False) for x in raw_parts_a]
        prompt_id = tokenizer.num_tokens

        def encode_input(raw_parts):
            parts = []
            for x, s in raw_parts:
                if isinstance(x, str):
                    x = tokenizer.EncodeAsIds(x)
                elif isinstance(x, int):
                    x = [prompt_id] * x
                else:
                    pass
                parts.append((x, s))
            return parts

        parts_a = encode_input(raw_parts_a)
        parts_b = None
        if raw_parts_b:
            raw_parts_b = [x if isinstance(x, tuple) else (
                x, False) for x in raw_parts_b]
            parts_b = encode_input(raw_parts_b)

        answers = self.get_answers(example)
        if example.label is not None:
            label = self.label_list.index(example.label)
        else:
            label = 0

        ids_list, positions_list, sep_list, mask_list, target_list, prompt_list = [], [], [], [], [], []
        segment_id_list = []

        for idx, answer in enumerate(answers):
            this_parts_a, this_parts_b = copy.deepcopy(
                parts_a), copy.deepcopy(parts_b)
            answer_ids = ids = tokenizer.EncodeAsIds(answer).tokenization
            answer_ids = answer_ids + [tokenizer.get_command('eop').Id]
            self.num_truncated += self.truncate(this_parts_a, this_parts_b, answer_ids,
                                                max_length=self.max_seq_length)
            tokens_a = [token_id for part,
                        _ in this_parts_a for token_id in part]
            tokens_b = [
                token_id for part, _ in this_parts_b for token_id in part] if parts_b else None
            segments = [answer_ids]
            for segment in segments:
                data = build_input_from_ids(tokens_a, tokens_b, segment, self.max_seq_length,
                                            self.tokenizer,
                                            add_cls=True, add_sep=False, add_piece=True,
                                            mask_id=self.mask_id)
                ids, types, paddings, position_ids, sep, target_ids, loss_masks = data
                prompt_pos = [idx for idx, token in enumerate(
                    ids) if token == prompt_id]
                ids = [idx if idx != prompt_id else 0 for idx in ids]
                prompt_list.append(prompt_pos)
                ids_list.append(ids)
                positions_list.append(position_ids)
                sep_list.append(sep)
                target_list.append(target_ids)
                mask_list.append(loss_masks)
                if self.mask in tokens_a:
                    mask_pos = tokens_a.index(self.mask)
                    tokens_a = tokens_a[:mask_pos] + \
                        segment + tokens_a[mask_pos:]
                else:
                    mask_pos = tokens_b.index(self.mask)
                    tokens_b = tokens_b[:mask_pos] + \
                        segment + tokens_b[mask_pos:]
        segment_id_list = segment_id_list if segment_id_list else None
        sample = build_sample(ids_list, positions=positions_list, masks=sep_list, label=label,
                              logit_mask=mask_list, target=target_list,
                              unique_id=example.guid, segment_ids=segment_id_list, prompt_ids=prompt_list,answer_ids=example.answer_idx)
        return sample

    def get_parts(self, example: InputExample) -> FilledPattern:
        premise = self.shortenable(example.text_a)

        assert '@placeholder' in example.text_b, f'question "{example.text_b}" does not contain a @placeholder token'
        question_a, question_b = example.text_b.split('@placeholder')
        return [premise, " " + question_a.rstrip(), [self.mask], question_b], []

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    @property
    def is_multi_token(self):
        return True

    def get_answers(self, example: InputExample):
        choices = example.meta['candidates']
        choices = [" " + choice for choice in choices]
        return choices

    def verbalize(self, label) -> List[str]:
        return []

    @property
    def mask(self) -> str:
        """Return the underlying LM's mask token"""
        return self.tokenizer.get_command('MASK').Id

    @property
    def mask_id(self) -> int:
        """Return the underlying LM's mask id"""
        return self.tokenizer.get_command('MASK').Id

    def truncate(self, parts_a: List[Tuple[List[int], bool]], parts_b: List[Tuple[List[int], bool]], answer: List[int],
                 max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        if answer:
            total_len += len(answer)
        total_len += num_special_tokens_to_add(
            parts_a, parts_b, answer, add_cls=True, add_sep=False, add_piece=True)
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return False

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)
        return True

    @staticmethod
    def _seq_length(parts: List[Tuple[List[int], bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[List[int], bool]]):
        last_idx = max(idx for idx, (seq, shortenable)
                       in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])


def num_special_tokens_to_add(text_a_ids, text_b_ids, answer_ids, add_cls, add_sep, add_piece, add_eos=True):
    num_tokens = 0
    if add_cls:
        num_tokens += 1
    if text_b_ids and add_sep:
        num_tokens += 1
    if add_eos:
        num_tokens += 1
    if not answer_ids and add_piece:
        num_tokens += 1
    return num_tokens


def build_input_from_ids(text_a_ids, text_b_ids, answer_ids, max_seq_length, tokenizer, add_cls=True,
                         add_sep=False, add_piece=False, add_eos=True, mask_id=None):
    if mask_id is None:
        mask_id = tokenizer.get_command('MASK').Id
    eos_id = tokenizer.get_command('eos').Id
    cls_id = tokenizer.get_command('ENC').Id
    sep_id = tokenizer.get_command('sep').Id
    ids = []
    types = []
    paddings = []
    # CLS
    if add_cls:
        ids.append(cls_id)
        types.append(0)
        paddings.append(1)
    # A
    len_text_a = len(text_a_ids)
    ids.extend(text_a_ids)
    types.extend([0] * len_text_a)
    paddings.extend([1] * len_text_a)
    # B
    if text_b_ids is not None:
        # SEP
        if add_sep:
            ids.append(sep_id)
            types.append(0)
            paddings.append(1)
        len_text_b = len(text_b_ids)
        ids.extend(text_b_ids)
        types.extend([1] * len_text_b)
        paddings.extend([1] * len_text_b)
    eos_length = 1 if add_eos else 0
    # Cap the size.
    if len(ids) >= max_seq_length - eos_length:
        max_seq_length_m1 = max_seq_length - 1
        ids = ids[0:max_seq_length_m1]
        types = types[0:max_seq_length_m1]
        paddings = paddings[0:max_seq_length_m1]
    end_type = 0 if text_b_ids is None else 1
    if add_eos:
        ids.append(eos_id)
        types.append(end_type)
        paddings.append(1)
    sep = len(ids)
    target_ids = [0] * len(ids)
    loss_masks = [0] * len(ids)
    position_ids = list(range(len(ids)))
    block_position_ids = [0] * len(ids)
    # Piece
    if add_piece or answer_ids is not None:
        sop_id = tokenizer.get_command('sop').Id
        mask_position = ids.index(mask_id)
        ids.append(sop_id)
        types.append(end_type)
        paddings.append(1)
        position_ids.append(mask_position)
        block_position_ids.append(1)
        if answer_ids is not None:
            len_answer = len(answer_ids)
            ids.extend(answer_ids[:-1])
            types.extend([end_type] * (len_answer - 1))
            paddings.extend([1] * (len_answer - 1))
            position_ids.extend([mask_position] * (len_answer - 1))
            block_position_ids.extend(range(2, len(answer_ids) + 1))
            target_ids.extend(answer_ids)
            loss_masks.extend([1] * len(answer_ids))
        else:
            target_ids.append(0)
            loss_masks.append(1)
    # Padding.
    padding_length = max_seq_length - len(ids)
    if padding_length > 0:
        ids.extend([eos_id] * padding_length)
        types.extend([eos_id] * padding_length)
        paddings.extend([0] * padding_length)
        position_ids.extend([0] * padding_length)
        block_position_ids.extend([0] * padding_length)
        target_ids.extend([0] * padding_length)
        loss_masks.extend([0] * padding_length)
    position_ids = [position_ids, block_position_ids]
    return ids, types, paddings, position_ids, sep, target_ids, loss_masks


def build_sample(ids, types=None, paddings=None, positions=None, masks=None, label=None, unique_id=None, target=None,
                 logit_mask=None, segment_ids=None, prompt_ids=None,answer_ids=None):
    """Convert to numpy and return a sample consumed by the batch producer."""

    ids_np = np.array(ids, dtype=np.int64)
    sample = {'text': ids_np, 'label': int(label),"answer_idx":answer_ids}
    if types is not None:
        types_np = np.array(types, dtype=np.int64)
        sample['types'] = types_np
    if paddings is not None:
        paddings_np = np.array(paddings, dtype=np.int64)
        sample['padding_mask'] = paddings_np
    if positions is not None:
        positions_np = np.array(positions, dtype=np.int64)
        sample['position'] = positions_np
    if masks is not None:
        masks_np = np.array(masks, dtype=np.int64)
        sample['mask'] = masks_np
    if target is not None:
        target_np = np.array(target, dtype=np.int64)
        sample['target'] = target_np
    if logit_mask is not None:
        logit_mask_np = np.array(logit_mask, dtype=np.int64)
        sample['logit_mask'] = logit_mask_np
    if segment_ids is not None:
        segment_ids = np.array(segment_ids, dtype=np.int64)
        sample['segment_id'] = segment_ids
    if prompt_ids is not None:
        prompt_ids = np.array(prompt_ids, dtype=np.int64)
        sample['prompt_pos'] = prompt_ids
    if unique_id is not None:
        sample['uid'] = unique_id
    return sample
