from pyvi import ViTokenizer
from tqdm import tqdm

import os
import re
import random as rand
import logging
import json

logger = logging.getLogger()
log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_format)
logger.handlers = [console_handler]


rand.seed(10)


class DataHelper:
    def __init__(self,
                 separator="\t",
                 fields=None,
                 word_segment=False,
                 use_char=False,
                 skip_header=True,
                 sort_batch=True,
                 min_pad=True,
                 padding_token="<PAD>",
                 start_token="<SOS>",
                 end_token="<EOS>",
                 unknown_token="<UNK>",
                 prepocess=None,
                 tokenizer=None):

        if fields is None:
            fields = ["TEXT", "LABEL"]

        if prepocess is None or not callable(prepocess):
            logger.warning("WARNING: Use default preprocess function!!!!")
            self.preprocess = self.__preprocess

        if tokenizer is None or not callable(prepocess):
            logger.warning("WARNING: Use default tokenizer function!!!!")
            self.tokenizer = self.__word_tokenizer

        self.padding_token = padding_token
        self.start_token = start_token
        self.end_token = end_token
        self.unk_token = unknown_token

        self.min_pad = min_pad
        self.skip_header = skip_header
        self.sort_batch = sort_batch
        self.use_char = use_char
        self.word_vocab = [self.padding_token, self.start_token, self.end_token, self.unk_token]
        self.label_vocab = []
        self.char_vocab = [self.padding_token, self.start_token, self.end_token, self.unk_token]
        self.separator = separator
        self.fields = fields
        self.word_segment = word_segment

        self.raw_examples = {"train": [], "valid": [], "test": []}
        self.raw_labels = {"train": [], "valid": [], "test": []}

        self.id_examples = {"train": [], "valid": [], "test": []}
        self.id_labels = {"train": [], "valid": [], "test": []}

        self.batches = []
        self.num_of_batch = 0

    def __word_tokenizer(self, text):
        text = ViTokenizer.tokenize(text)
        text = self.preprocess(text)
        if not self.word_segment:
            text = re.sub('_', ' ', text)
        return text.strip().split()

    @staticmethod
    def __preprocess(text):
        return text

    def read_csv(self, train_file=None, valid_file=None, test_file=None):
        logger.info("Loadding dataset ...")
        word_set = set()
        char_set = set()
        label_set = set()

        if train_file is not None:
            logger.info("Load train dataset from: %s", train_file)
            word_set_tmp, char_set_tmp, label_set_tmp = self.__data_loader(train_file, "train")
            word_set.update(word_set_tmp)
            char_set.update(char_set_tmp)
            label_set.update(label_set_tmp)
        if valid_file is not None:
            logger.log("Load validation dataset: from %s", valid_file)
            word_set_tmp, char_set_tmp, label_set_tmp = self.__data_loader(valid_file, "valid")
            word_set.update(word_set_tmp)
            char_set.update(char_set_tmp)
            label_set.update(label_set_tmp)
        if test_file is not None:
            logger.log("Load test dataset: from %s", test_file)
            word_set_tmp, char_set_tmp, label_set_tmp = self.__data_loader(valid_file, "test")
            word_set.update(word_set_tmp)
            char_set.update(char_set_tmp)
            label_set.update(label_set_tmp)

        self.word_vocab.extend(list(word_set))
        self.char_vocab.extend(list(char_set))
        self.label_vocab.extend(list(label_set))

        self.__data_id_convert()
        logger.info("Load dataset finished!")
        logger.info("")
        logger.info("#"*10 + "Statistical Data" + "#"*10)
        logger.info("Dataset:")
        logger.info("   Number of training examples: {}".format(len(self.raw_labels["train"])))
        logger.info("   Number of vadilation examples: {}".format(len(self.raw_labels["valid"])))
        logger.info("   Number of testing examples: {}".format(len(self.raw_labels["test"])))
        logger.info("Vocabulary:")
        logger.info("   Word vocabulary size: {}".format(len(self.word_vocab)))
        logger.info("   Character vocabulary size: {}".format(len(self.char_vocab)))
        logger.info("   Target vocabulary size: {}".format(len(self.label_vocab)))
        logger.info("#"*35)
        logger.info("")

    def iterator(self, batch_size=64, shuffle=True, type_data="train"):
        logger.info("Build {} Iterator ...".format(type_data))
        batch = []
        max_seq_len = 0
        max_word_len = 0

        if shuffle and self.min_pad:
            logger.warning("WARNING: Can not shuffle when min_pad == True. Shuffle was set 'Fasle'!")
            shuffle = False

        if shuffle:
            zipper = list(zip(self.id_examples[type_data], self.id_labels[type_data]))
            rand.shuffle(zipper)
            examples, labels = zip(*zipper)
        elif self.min_pad:
            zipper = list(zip(self.id_examples[type_data], self.id_labels[type_data]))
            zipper.sort(key=lambda x: len(x[0][0]), reverse=True)
            examples, labels = zip(*zipper)
        else:
            examples = self.id_examples[type_data]
            labels = self.id_labels[type_data]

        for idx in range(len(examples)):
            id_seq = examples[idx][0]
            id_chars = examples[idx][1]
            label = labels[idx]
            max_seq_len = len(id_seq) if max_seq_len < len(id_seq) else max_seq_len

            length = len(max(id_chars, key=len))
            max_word_len = length if max_word_len < length else max_word_len
            batch.append((id_seq, id_chars, label))

            if len(batch) == batch_size:
                self.batches.append(self.__batch_padding(batch, max_seq_len, max_word_len))
                max_seq_len = 0
                max_word_len = 0
                batch = []
        if not len(batch) == 0:
            self.batches.append(self.__batch_padding(batch, max_seq_len, max_word_len))
        self.num_of_batch = len(self.batches)
        logger.info("Build Iterator finished!")
        logger.info("")
        logger.info("#" * 10 + "Statistical Iterator" + "#" * 10)
        logger.info("Number of batch: {}".format(self.num_of_batch))
        logger.info("Number of example in each batch: {}".format(batch_size))
        logger.info("#" * 35)
        logger.info("")

        return self.batches

    def save_vocab_to_json(self, file_path):
        fo = open(file_path, "w", encoding="utf-8")
        vocabs = (self.word_vocab, self.char_vocab, self.label_vocab)
        json.dump(vocabs, fo)
        logger.info("Vocabulary was saved in '{}'!!!!".format(file_path))
        fo.close()

    def load_vocab_from_json(self, file_path):
        if not os.path.isfile(file_path):
            logger.error("ERROR: '{}' is not file json format or not exit!!!!".format(file_path))
        fi = open(file_path, "r", encoding="utf-8")
        self.word_vocab, self.char_vocab, self.label_vocab = json.load(fi)
        logger.info("Load vocabulary from '{}'  successful!!!!".format(file_path))

    def __batch_padding(self, batch, max_seg_len, max_word_len):
        word_list = []
        char_list = []
        label_list = []
        seq_lengths = []

        for example in batch:
            label_list.append(example[-1])

            seq_len = len(example[0])
            num_of_seq_pad = max_seg_len-seq_len

            seq_lengths.append(seq_len)
            example[0].extend([self.word_vocab.index(self.padding_token)] * num_of_seq_pad)
            word_list.append(example[0])

            char_seq = []
            for word in example[1]:
                word.extend([self.char_vocab.index(self.padding_token)] * (max_word_len-len(word)))
                char_seq.append(word)
            char_seq.extend([[self.char_vocab.index(self.padding_token)] * max_word_len]*num_of_seq_pad)
            char_list.append(char_seq)

        if self.sort_batch and not self.min_pad:
            return self.__sort_batch(word_list, char_list, label_list, seq_lengths)

        return word_list, char_list, label_list, seq_lengths

    @staticmethod
    def __sort_batch(word_list, char_list, label_list, seq_lengths):
        sorted_index = [i[0] for i in sorted(enumerate(seq_lengths), key=lambda x: x[1], reverse=True)]

        sorted_word_list = [word_list[i] for i in sorted_index]
        sorted_char_list = [char_list[i] for i in sorted_index]
        sorted_label_list = [label_list[i] for i in sorted_index]
        sorted_seq_lengths = [seq_lengths[i] for i in sorted_index]

        return sorted_word_list, sorted_char_list, sorted_label_list, sorted_seq_lengths

    def __data_loader(self, file, type_data):
        word_set = set()
        char_set = set()
        label_set = set()
        lines = open(file, 'r', encoding='utf-8').readlines()
        if self.skip_header:
            lines.pop(0)

        for line in tqdm(lines):
            contents = line.split(self.separator)
            assert len(contents) == len(self.fields)
            for idx, content in enumerate(contents):
                if self.fields[idx] is None:
                    continue
                elif self.fields[idx] == "LABEL":
                    content = content.strip()
                    label_set.add(content)
                    self.raw_labels[type_data].append(content)
                else:
                    content = self.tokenizer(content)
                    chars_list = []
                    for word in content:
                        word_set.add(word)
                        char_set.update(list(word))
                        chars_list.append(list(word))
                    self.raw_examples[type_data].append((content, chars_list))
        return word_set, char_set, label_set

    def __data_id_convert(self):
        for idx in range(len(self.raw_examples["train"])):
            example = self.raw_examples["train"][idx]

            self.id_examples["train"].append(([self.word_vocab.index(w) for w in example[0]],
                                              [[self.char_vocab.index(c) for c in w] for w in example[1]]))
            self.id_labels["train"].append(self.label_vocab.index(self.raw_labels["train"][idx]))

        for idx in range(len(self.raw_examples["valid"])):
            example = self.raw_examples["valid"][idx]
            self.id_examples["valid"].append(([self.word_vocab.index(w) for w in example[0]],
                                              [[self.char_vocab.index(c) for c in w] for w in example[1]]))
            self.id_labels["valid"].append(self.label_vocab.index(self.raw_labels["valid"][idx]))

        for idx in range(len(self.raw_examples["test"])):
            example = self.raw_examples["test"][idx]
            self.id_examples["test"].append(([self.word_vocab.index(w) for w in example[0]],
                                             [[self.char_vocab.index(c) for c in w] for w in example[1]]))
            self.id_labels["test"].append(self.label_vocab.index(self.raw_labels["test"][idx]))


"""
data = DataHelper()
data.read_csv("debug.csv")
batches = data.iterator(shuffle=True, batch_size=12, type="train")
data.save_vocab_to_json("vocab.json")
data.load_vocab_from_json("vocab.json")
"""
