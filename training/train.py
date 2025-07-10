# This file aims to train a PagPassGPT with checkpoint support.

import sys
import time
import argparse
import os
import torch

sys.path.append("/kaggle/input/pagpassgpt/tokenizer/")
from char_tokenizer import CharTokenizer
from transformers import DataCollatorForLanguageModeling
from datasets import load_dataset
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, EarlyStoppingCallback

os.environ["WANDB_DISABLED"] = "true"

parser = argparse.ArgumentParser()

# File paths
parser.add_argument("--dataset_path", type=str, required=True, help="Path of preprocessed train dataset")
parser.add_argument("--vocabfile_path", type=str, default="./tokenizer/vocab.json", help="Path of vocab file")
parser.add_argument("--model_path", type=str, default="./model/", help="Directory to save model")
parser.add_argument("--log_path", type=str, default="./log/", help="Directory of log")

# Environment
parser.add_argument("--random_seed", type=int, default=42, help="Random seed")
parser.add_argument("--num_processer", type=int, default=1, help="Num of processors")

# Model params
parser.add_argument("--input_size", type=int, default=32, help="Should be larger than (2*max len + 3)")
parser.add_argument("--embed_size", type=int, default=384, help="Embedding size")
parser.add_argument("--layer_num", type=int, default=12, help="Num of transformer layers")
parser.add_argument("--head_num", type=int, default=8, help="Num of attention heads")

# Training params
parser.add_argument("--epoch_num", type=int, default=30, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
parser.add_argument("--eval_step", type=int, default=2000, help="Eval every n steps")
parser.add_argument("--save_step", type=int, default=6000, help="Save every n steps")
parser.add_argument("--early_stop", type=int, default=3, help="Early stopping patience")
parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from")

args = parser.parse_args()

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# Paths
train_dataset_path = args.dataset_path
vocab_file = args.vocabfile_path
model_output_dir = args.model_path
log_dir = args.log_path
resume_ckpt = args.resume_from_checkpoint

# Params
random_seed = args.random_seed
num_processer = args.num_processer
input_size = args.input_size
embed_size = args.embed_size
layer_num = args.layer_num
head_num = args.head_num
epoch_num = args.epoch_num
batch_size = args.batch_size
eval_step = args.eval_step
save_step = args.save_step
early_stop = args.early_stop

# Tokenizer
print('[INFO] Loading tokenizer.')
tokenizer = CharTokenizer(
    vocab_file=vocab_file,
    bos_token="<BOS>",
    eos_token="<EOS>",
    sep_token="<SEP>",
    unk_token="<UNK>",
    pad_token="<PAD>",
)

# Dataset
print('[INFO] Loading dataset.')
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
raw_dataset = load_dataset('text', data_files=train_dataset_path, num_proc=num_processer, split='train')
tokenized_dataset = raw_dataset.map(
    lambda examples: tokenizer(examples['text'], max_len=input_size, padding=True),
    batched=True
)

print('[INFO] Splitting dataset into training and validation sets.')
split_dataset = tokenized_dataset.train_test_split(test_size=0.125)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Model
if resume_ckpt and os.path.exists(resume_ckpt):
    print(f'[INFO] Resuming training from checkpoint: {resume_ckpt}')
    model = GPT2LMHeadModel.from_pretrained(resume_ckpt)
else:
    print(f'[INFO] Training new model from scratch.')
    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=input_size,
        n_embd=embed_size,
        n_layer=layer_num,
        n_head=head_num,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
    )
    model = GPT2LMHeadModel(config=config)

model.to(device)

print(f"[INFO] Model Parameters: {model.num_parameters()}")

# Training Arguments
print('[INFO] Preparing training arguments.')
training_args = TrainingArguments(
    output_dir=model_output_dir,
    overwrite_output_dir=True,
    num_train_epochs=epoch_num,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_steps=eval_step,
    save_steps=save_step,
    save_strategy='steps',
    eval_strategy='steps',
    logging_strategy='steps',
    logging_steps=eval_step,
    prediction_loss_only=True,
    disable_tqdm=True,
    report_to=None,
    logging_dir=os.path.join(log_dir, time.strftime("%Y%m%d-%H%M", time.localtime())),
    seed=random_seed,
    metric_for_best_model='eval_loss',
    load_best_model_at_end=True,
    save_total_limit=1,
    no_cuda=not torch.cuda.is_available()
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Train
print('*' * 30)
print('[INFO] Training begins.')
trainer.train(resume_from_checkpoint=resume_ckpt)
print('*' * 30)
print('[INFO] Training complete.')

# Save final model
final_save_path = os.path.join(model_output_dir, "last-step")
trainer.save_model(final_save_path)
print(f'[INFO] Final model saved to {final_save_path}')
