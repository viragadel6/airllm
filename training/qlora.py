# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from os.path import exists, isdir, join
from typing import Dict, Optional, Sequence

import bitsandbytes as bnb
import evaluate
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import Dataset, DatasetDict, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Seq2SeqTrainer, set_seed
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("./qlora_logs.log"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
COMPLETED_FILE = "completed"
CHECKPOINT_PREFIX = f"{PREFIX_CHECKPOINT_DIR}-"
DEBUG_MODE_SAMPLES = 200
SAMPLE_MAX_NEW_TOKENS = 70
ANSWER_LETTERS = ("A", "B", "C", "D")
PROMPT_COLUMNS = ("input", "output")

SAMPLE_PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response: "
)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="EleutherAI/pythia-12b"
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )


@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|alpaca-clean|chip2|self-instruct|hh-rlhf|oasst1|chinese-vicuna]"}
    )


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    mmlu_split: Optional[str] = field(
        default='eval',
        metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default='mmlu-fs',
        metadata={"help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={"help": "If set, only evaluates on `max_mmlu_samples` of the MMLU dataset."}
    )
    mmlu_source_max_len: int = field(
        default=2048,
        metadata={"help": "Maximum source sequence length for mmlu."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(
        default=False,
        metadata={"help": "Use 8-bit adam."}
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    lora_r: int = field(
        default=64,
        metadata={"help": "Lora R dimension."}
    )
    lora_alpha: float = field(
        default=16,
        metadata={"help": "Lora alpha."}
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Lora dropout."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before performing an optimizer step'})
    max_steps: int = field(default=10000, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'})
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learning rate'})
    remove_unused_columns: bool = field(default=False, metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='steps', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    sample_generate: bool = field(default=False, metadata={"help": 'If do sample generation on evaluation.'})
    debug_mode: bool = field(default=False, metadata={"help": 'debug mode sample 200 train/eval samples for validation'})


@dataclass
class GenerationArguments:
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    temperature: Optional[float] = field(default=None)
    top_k: Optional[int] = field(default=None)
    top_p: Optional[float] = field(default=None)
    typical_p: Optional[float] = field(default=None)
    diversity_penalty: Optional[float] = field(default=None)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def get_answer_token_ids(tokenizer):
    answer_token_ids = []
    for letter in ANSWER_LETTERS:
        token_ids = tokenizer(letter, add_special_tokens=False).input_ids
        if len(token_ids) != 1:
            raise ValueError(
                f"Tokenizing the answer letter '{letter}' produced {len(token_ids)} tokens; "
                "MMLU evaluation requires every answer letter to be a single token."
            )
        answer_token_ids.append(token_ids[0])
    return answer_token_ids


def get_device(model):
    device = getattr(model, 'device', None)
    if device is None:
        device = next(model.parameters()).device
    return device


def mark_training_completed(output_dir):
    completed_file = join(output_dir, COMPLETED_FILE)
    with open(completed_file, 'a'):
        os.utime(completed_file, None)


class SampleGenerateCallback(transformers.TrainerCallback):
    sample_inputs = [
        '用一句话描述地球为什么是独一无二的。',
        '中国是否应该推出刺激政策救楼市？',
        '如何更好地融入新工作圈子'
    ]

    def on_evaluate(self, args, state, control, **kwargs):
        logger.info('on_evaluate in SampleGenerateCallback...')
        model = kwargs.get('model')
        tokenizer = kwargs.get('processing_class')
        if model is None or tokenizer is None:
            logger.info('model or tokenizer not found in kwargs, skipping')
            return control

        device = get_device(model)
        use_cache = model.config.use_cache
        gradient_checkpointing = getattr(model, 'is_gradient_checkpointing', False)
        model.config.use_cache = True
        if gradient_checkpointing:
            model.gradient_checkpointing_disable()
        try:
            for sample_input in self.sample_inputs:
                prompt = SAMPLE_PROMPT.format(instruction=sample_input)
                logger.info(f"sample input: {prompt}")
                input_ids = tokenizer(prompt, return_tensors="pt")['input_ids'].to(device)
                generation_output = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=SAMPLE_MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True,
                )
                logger.info(f"sample output: {tokenizer.decode(generation_output[0])}")
        finally:
            model.config.use_cache = use_cache
            if gradient_checkpointing:
                model.gradient_checkpointing_enable()
        return control


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_adapter(self, model, output_dir):
        logger.info(f"Saving PEFT checkpoint to {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        pytorch_model_path = join(output_dir, 'pytorch_model.bin')
        if exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get('model')
        if model is None:
            logger.info('model not found in kwargs, skipping')
            return control
        if not getattr(args, 'should_save', True):
            return control
        self.save_adapter(model, join(args.output_dir, f"{CHECKPOINT_PREFIX}{state.global_step}"))
        return control

    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs.get('model')
        if model is None:
            logger.info('model not found in kwargs, skipping')
            return control
        if not getattr(args, 'should_save', True):
            return control
        self.save_adapter(model, args.output_dir)
        mark_training_completed(args.output_dir)
        logger.info('Training completed, the final adapter was saved.')
        return control


def get_quantization_config(args, compute_dtype):
    if args.bits not in (4, 8):
        return None
    return BitsAndBytesConfig(
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=args.double_quant,
        bnb_4bit_quant_type=args.quant_type,
    )


def get_adapter_path(checkpoint_dir):
    if exists(join(checkpoint_dir, 'adapter_config.json')):
        return checkpoint_dir
    nested_checkpoint_dir = join(checkpoint_dir, 'adapter_model')
    if exists(join(nested_checkpoint_dir, 'adapter_config.json')):
        return nested_checkpoint_dir
    return checkpoint_dir


def get_token_from_id(tokenizer, token_id, fallback_token):
    if token_id is None:
        return fallback_token
    token = tokenizer.convert_ids_to_tokens(token_id)
    if token is None:
        return fallback_token
    return token


def add_llama_special_tokens(tokenizer, model):
    model_config = model.config
    unk_token_id = getattr(model_config, 'unk_token_id', None)
    if unk_token_id is None:
        unk_token_id = tokenizer.unk_token_id
    special_tokens_dict = {}
    for token_name, token_id, fallback_token in (
        ('eos_token', model_config.eos_token_id, tokenizer.eos_token),
        ('bos_token', model_config.bos_token_id, tokenizer.bos_token),
        ('unk_token', unk_token_id, tokenizer.unk_token),
    ):
        token = get_token_from_id(tokenizer, token_id, fallback_token)
        if token is not None:
            special_tokens_dict[token_name] = token
    if special_tokens_dict:
        tokenizer.add_special_tokens(special_tokens_dict)


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def build_tokenizer(args):
    return AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False,
    )


def get_accelerate_model(args, tokenizer, checkpoint_dir):
    n_gpus = torch.cuda.device_count()
    max_memory = {gpu: f'{args.max_memory_MB}MB' for gpu in range(n_gpus)}

    if args.full_finetune:
        assert args.bits in [16, 32]

    logger.info(f'loading base model {args.model_name_or_path}...')
    compute_dtype = torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        quantization_config=get_quantization_config(args, compute_dtype),
        device_map='auto',
        max_memory=max_memory,
        torch_dtype=compute_dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            logger.info('=' * 80)
            logger.info('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            logger.info('=' * 80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype = compute_dtype
    model.config.use_cache = False

    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    if not args.full_finetune:
        if args.bits in (4, 8):
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
        else:
            for _, parameter in model.named_parameters():
                parameter.requires_grad = False

    if args.gradient_checkpointing and not getattr(model, 'is_gradient_checkpointing', False):
        model.gradient_checkpointing_enable()

    if not args.full_finetune:
        if checkpoint_dir is not None:
            adapter_path = get_adapter_path(checkpoint_dir)
            logger.info(f"Loading adapters from checkpoint {adapter_path}.")
            model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        else:
            logger.info('adding LoRA modules...')
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model


def print_trainable_parameters(args, model):
    trainable_params = 0
    all_param = 0
    for _, parameter in model.named_parameters():
        all_param += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()
    total_params_fp32_equivalent = all_param * 2 if args.bits == 4 else all_param
    trainable_percentage = 100 * trainable_params / total_params_fp32_equivalent if total_params_fp32_equivalent else 0.0
    logger.info(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} ({total_params_fp32_equivalent} fp32-equivalent) || "
        f"trainable%: {trainable_percentage:.4f}"
    )


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        bos_token = self.tokenizer.bos_token if self.tokenizer.bos_token is not None else ''
        eos_token = self.tokenizer.eos_token if self.tokenizer.eos_token is not None else ''
        sources = [f"{bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{eos_token}" for example in instances]
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict


ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}


def extract_alpaca_dataset(example):
    instruction = example.get("instruction", "")
    example_input = example.get("input", "")
    if example_input != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(instruction=instruction, input=example_input)}


def local_dataset(dataset_name):
    if dataset_name.endswith(('.json', '.jsonl')):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith('.csv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith('.tsv'):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter='\t'))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")

    split_dataset = full_dataset.train_test_split(test_size=0.1)
    return split_dataset


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == 'alpaca-clean':
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == 'chip2':
            return load_dataset("laion/OIG", data_files='unified_chip2.jsonl')
        elif dataset_name == 'self-instruct':
            return load_dataset("yizhongw/self_instruct", name='self_instruct')
        elif dataset_name == 'hh-rlhf':
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == 'longform':
            return load_dataset("akoksal/LongForm")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == 'vicuna':
            raise NotImplementedError("Vicuna data was not released.")
        elif dataset_name == 'chinese-vicuna':
            return load_dataset("Chinese-Vicuna/guanaco_belle_merge_v1.0")
        else:
            if not os.path.exists(dataset_name):
                raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")
            args.dataset_format = args.dataset_format if args.dataset_format else "alpaca"
            try:
                return local_dataset(dataset_name)
            except Exception as exception:
                raise ValueError(f"Error loading dataset from {dataset_name}: {exception}") from exception

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format in ('alpaca', 'alpaca-clean', 'chinese-vicuna') or
            (dataset_format is None and args.dataset in ('alpaca', 'alpaca-clean', 'chinese-vicuna'))
        ):
            dataset = dataset.map(extract_alpaca_dataset)
        elif dataset_format == 'chip2' or (dataset_format is None and args.dataset == 'chip2'):
            dataset = dataset.map(lambda x: {
                'input': x['text'].split('\n<bot>: ')[0].replace('<human>: ', ''),
                'output': x['text'].split('\n<bot>: ')[1],
            })
        elif dataset_format == 'self-instruct' or (dataset_format is None and args.dataset == 'self-instruct'):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == 'hh-rlhf' or (dataset_format is None and args.dataset == 'hh-rlhf'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['chosen']
            })
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })

        formatted_splits = {}
        for split_name, split_dataset in dataset.items():
            unused_columns = [column for column in split_dataset.column_names if column not in PROMPT_COLUMNS]
            if unused_columns:
                split_dataset = split_dataset.remove_columns(unused_columns)
            formatted_splits[split_name] = split_dataset
        return DatasetDict(formatted_splits)

    dataset = load_data(args.dataset)
    if args.debug_mode:
        for split_name in list(dataset.keys()):
            dataset[split_name] = dataset[split_name].select(range(min(DEBUG_MODE_SAMPLES, len(dataset[split_name]))))
    dataset = format_dataset(dataset, args.dataset_format)

    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        else:
            logger.info('Splitting train dataset in train and validation according to `eval_dataset_size`')
            train_dataset_size = len(dataset['train'])
            if train_dataset_size < 2:
                raise ValueError(
                    f"The train dataset must contain at least 2 examples to split off an eval dataset but contains "
                    f"{train_dataset_size} examples."
                )
            test_size = args.eval_dataset_size
            if test_size >= train_dataset_size:
                test_size = max(1, int(train_dataset_size * 0.1))
                logger.info(
                    f"Using an eval dataset size of {test_size} because the train dataset only contains "
                    f"{train_dataset_size} examples."
                )
            dataset = dataset['train'].train_test_split(
                test_size=test_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=args.source_max_len,
        target_max_len=args.target_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )


class MMLUEvalCallback(transformers.TrainerCallback):
    def __init__(self, trainer, tokenizer, eval_dataset, mmlu_split, mmlu_source_max_len):
        self.trainer = trainer
        self.eval_dataset = eval_dataset
        self.mmlu_split = mmlu_split
        self.mmlu_source_max_len = mmlu_source_max_len
        self.answer_token_ids = get_answer_token_ids(tokenizer)
        self.accuracy = evaluate.load("accuracy")

    def on_evaluate(self, args, state, control, model, **kwargs):
        data_loader = self.trainer.get_eval_dataloader(self.eval_dataset)
        data_collator = self.trainer.data_collator
        collator_source_max_len = data_collator.source_max_len
        collator_predict_with_generate = data_collator.predict_with_generate
        trainer_predict_with_generate = self.trainer.args.predict_with_generate
        if trainer_predict_with_generate:
            logger.info('MMLU evaluation requires logits, disabling predict_with_generate for the MMLU evaluation.')
        data_collator.source_max_len = self.mmlu_source_max_len
        data_collator.predict_with_generate = False
        self.trainer.args.predict_with_generate = False
        was_training = model.training
        model.eval()

        predictions, references, sample_subjects = [], [], []
        loss_mmlu = 0.0
        subjects = self.eval_dataset['subject']
        sample_index = 0
        try:
            for batch in tqdm(data_loader, total=len(data_loader)):
                loss, logits, labels = self.trainer.prediction_step(
                    self.trainer.model, batch, prediction_loss_only=False
                )
                if loss is not None:
                    loss_mmlu += float(loss)
                for index, sample_logits in enumerate(logits):
                    answer_positions = (batch['labels'][index] != IGNORE_INDEX).nonzero().flatten()
                    if answer_positions.numel() == 0:
                        logger.warning(
                            f"Skipping MMLU sample {sample_index + index} because it has no answer tokens."
                        )
                        continue
                    answer_position = int(answer_positions[0])
                    answer_token_id = int(labels[index][answer_position])
                    if answer_position < 1:
                        logger.warning(
                            f"Skipping MMLU sample {sample_index + index} because it has no token before the answer."
                        )
                        continue
                    if answer_token_id not in self.answer_token_ids:
                        logger.warning(
                            f"Skipping MMLU sample {sample_index + index} because the answer token id "
                            f"{answer_token_id} is not one of {self.answer_token_ids}."
                        )
                        continue
                    answer_logits = sample_logits[answer_position - 1][self.answer_token_ids]
                    predictions.append(int(torch.argmax(answer_logits).item()))
                    references.append(self.answer_token_ids.index(answer_token_id))
                    sample_subjects.append(subjects[sample_index + index])
                sample_index += len(logits)
        finally:
            data_collator.source_max_len = collator_source_max_len
            data_collator.predict_with_generate = collator_predict_with_generate
            self.trainer.args.predict_with_generate = trainer_predict_with_generate
            model.train(was_training)

        results = {'mmlu_loss': loss_mmlu / max(len(data_loader), 1)}
        subject_predictions = {}
        subject_references = {}
        for subject, prediction, reference in zip(sample_subjects, predictions, references):
            subject_predictions.setdefault(subject, []).append(prediction)
            subject_references.setdefault(subject, []).append(reference)
        subject_scores = []
        for subject, subject_prediction in sorted(subject_predictions.items()):
            subject_score = self.accuracy.compute(
                references=subject_references[subject],
                predictions=subject_prediction
            )['accuracy']
            results[f'mmlu_{self.mmlu_split}_accuracy_{subject}'] = subject_score
            subject_scores.append(subject_score)
        results[f'mmlu_{self.mmlu_split}_accuracy'] = float(np.mean(subject_scores)) if subject_scores else 0.0
        self.trainer.log(results)
        return control


def build_mmlu_eval_callback(trainer, tokenizer, args):
    if args.mmlu_dataset == 'mmlu-zs':
        mmlu_dataset = load_dataset("json", data_files={
            'eval': 'data/mmlu/zero_shot_mmlu_val.json',
            'test': 'data/mmlu/zero_shot_mmlu_test.json',
        })
    elif args.mmlu_dataset in ('mmlu', 'mmlu-fs'):
        mmlu_dataset = load_dataset("json", data_files={
            'eval': 'data/mmlu/five_shot_mmlu_val.json',
            'test': 'data/mmlu/five_shot_mmlu_test.json',
        })
    else:
        raise ValueError(
            f"Unsupported MMLU dataset {args.mmlu_dataset}; options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."
        )
    if args.mmlu_split not in mmlu_dataset:
        raise ValueError(f"Unsupported MMLU split {args.mmlu_split} for dataset {args.mmlu_dataset}.")
    mmlu_eval_dataset = mmlu_dataset[args.mmlu_split]
    if args.max_mmlu_samples is not None:
        mmlu_eval_dataset = mmlu_eval_dataset.select(range(min(args.max_mmlu_samples, len(mmlu_eval_dataset))))
    return MMLUEvalCallback(trainer, tokenizer, mmlu_eval_dataset, args.mmlu_split, args.mmlu_source_max_len)


def get_last_checkpoint(checkpoint_dir):
    if not isdir(checkpoint_dir):
        return None, False
    if exists(join(checkpoint_dir, COMPLETED_FILE)):
        return None, True
    max_step = 0
    for filename in os.listdir(checkpoint_dir):
        if not filename.startswith(CHECKPOINT_PREFIX):
            continue
        if not isdir(join(checkpoint_dir, filename)):
            continue
        step = filename[len(CHECKPOINT_PREFIX):]
        if not step.isdigit():
            continue
        max_step = max(max_step, int(step))
    if max_step == 0:
        return None, False
    checkpoint_dir = join(checkpoint_dir, f"{CHECKPOINT_PREFIX}{max_step}")
    logger.info(f"Found a previous checkpoint at: {checkpoint_dir}")
    return checkpoint_dir, False


def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, _ = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    generation_config = transformers.GenerationConfig(**vars(generation_args))
    if generation_config.use_cache is None:
        generation_config.use_cache = True
    training_args.generation_config = generation_config
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    logger.info(f"args: {args}")

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        logger.info('Detected that training was already completed!')
        return

    set_seed(args.seed)

    tokenizer = build_tokenizer(args)
    model = get_accelerate_model(args, tokenizer, checkpoint_dir)
    print_trainable_parameters(args, model)
    logger.info('loaded model')

    if 'llama' in args.model_name_or_path.lower():
        logger.info('Adding special tokens.')
        add_llama_special_tokens(tokenizer, model)

    data_module = make_data_module(tokenizer=tokenizer, args=args)
    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        **{key: value for key, value in data_module.items() if key != 'predict_dataset'},
    )

    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback)
    if args.sample_generate:
        trainer.add_callback(SampleGenerateCallback)
    if args.do_mmlu_eval:
        trainer.add_callback(build_mmlu_eval_callback(trainer, tokenizer, args))

    dtypes = {}
    for _, parameter in model.named_parameters():
        if parameter.dtype not in dtypes:
            dtypes[parameter.dtype] = 0
        dtypes[parameter.dtype] += parameter.numel()
    total = sum(dtypes.values())
    for dtype, count in dtypes.items():
        percentage = 100 * count / total if total else 0.0
        logger.info(f"dtype {dtype}: {count} parameters ({percentage:.2f}%)")

    all_metrics = {"run_name": args.run_name}

    if args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)

    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'], metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = np.asarray(predictions)
        if predictions.dtype.kind in ('U', 'O', 'S'):
            decoded_predictions = [str(prediction) for prediction in predictions]
        else:
            if args.predict_with_generate:
                token_predictions = predictions
            else:
                token_predictions = np.argmax(predictions, axis=-1)
            token_predictions = np.where(token_predictions < 0, tokenizer.pad_token_id, token_predictions)
            decoded_predictions = tokenizer.batch_decode(
                token_predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for index, example in enumerate(data_module['predict_dataset']):
                example = dict(example)
                example['prediction_with_input'] = decoded_predictions[index].strip()
                example['prediction'] = decoded_predictions[index].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        logger.info(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if args.do_train or args.do_eval or args.do_predict:
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


if __name__ == "__main__":
    train()
