import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast, MBartConfig

from load_dataset import MyDataset

import argparse

def main():
    parser = argparse.ArgumentParser(description='training job')
    parser.add_argument('--data_dir',  type=str, help='Total epochs to train the model')
    parser.add_argument('--train_split', default="train",  type=str, help='Total epochs to train the model')
    parser.add_argument('--val_split', default="dev",  type=str, help='Total epochs to train the model')
    parser.add_argument('-s','--source_ln', default="SRC",  type=str, help='Total epochs to train the model')
    parser.add_argument('-t','--target_ln', default="TGT",  type=str, help='Total epochs to train the model')
    parser.add_argument('--pretrained_encoder_name',  type=str, help='Total epochs to train the model')
    parser.add_argument('--num_encoders', default=6, type=int, help='Total epochs to train the model')
    parser.add_argument('--num_decoders', default=6, type=int, help='Total epochs to train the model')
    parser.add_argument('--total_epochs', default=15, type=int, help='Total epochs to train the model')
    parser.add_argument('--max_len', default=210, type=int, help='Maximul length of source and target sentences')
    parser.add_argument('--save_dir',type=str, help="checkpoint dir path")
    parser.add_argument('--lr',type=float,default=1e-4, help="learning rate")
    parser.add_argument('--warmup_steps',type=int, help="warmup update steps")
    parser.add_argument('--batch_size', default=4, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--grad_accum_steps', default=1, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--total_save_limit', default=5, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--logging_dir', default="./logs", type=str, help='Input batch size on each device (default: 32)')
    parser.add_argument('--logging_steps', default=100, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--num_workers', default=1, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--clip_norm', default=1.0, type=float, help='Input batch size on each device (default: 32)')
    
    args = parser.parse_args()
    print(args.pretrained_encoder_name)

    train_source_file = f"{args.data_dir}/{args.train_split}.{args.source_ln}"
    train_target_file = f"{args.data_dir}/{args.train_split}.{args.target_ln}"
    
    eval_source_file = f"{args.data_dir}/{args.val_split}.{args.source_ln}"
    eval_target_file = f"{args.data_dir}/{args.val_split}.{args.target_ln}"
    
    tokenizer = MBart50TokenizerFast.from_pretrained(args.pretrained_encoder_name,src_lang="hi_IN",tgt_lang="en_XX")
    
    train_dataset = MyDataset(
        source_file = train_source_file, 
        target_file = train_target_file, 
        tokenizer = tokenizer,
        max_length = args.max_len
    )
    
    eval_dataset = MyDataset(
        source_file = eval_source_file, 
        target_file = eval_target_file, 
        tokenizer = tokenizer, 
        max_length = args.max_len
    )
    
    #config = MBartConfig.from_pretrained(args.pretrained_encoder_name,encoder_layers=6, decoder_layers=6)
    model = MBartForConditionalGeneration.from_pretrained("../mbart50_pretrained_loaded_6_en_6_de")
    for param in model.model.shared.parameters():
        param.requires_grad = False
    for param in model.model.encoder.parameters():
        param.requires_grad = False

    training_args = Seq2SeqTrainingArguments(
    output_dir=args.save_dir,                            # Directory where the model checkpoints and logs will be saved.
    num_train_epochs=args.total_epochs,                  # Total number of training epochs.
    gradient_accumulation_steps=args.grad_accum_steps,   # Number of update steps to accumulate gradients before performing a backward/update pass.
    evaluation_strategy='epoch',                         # Evaluation is performed at the end of each epoch.
    load_best_model_at_end=True,                         # Load and evaluate the best model at the end of training.
    metric_for_best_model='loss',                        # Metric to use to evaluate the best model.
    save_strategy='epoch',
    fp16=True,                                           # Enable mixed precision for faster training.
    fp16_opt_level='02',                                 # Optimization level for mixed precision training.                          # Save a checkpoint at the end of each epoch.
    save_total_limit=args.total_save_limit,              # Limit the total number of checkpoints.
    logging_dir=f"{args.save_dir}/{args.logging_dir}",                        # Directory where logs will be written.
    logging_steps=args.logging_steps,                    # Log every specified number of steps.
    report_to='tensorboard',                             # Send training logs to TensorBoard.
    dataloader_num_workers=args.num_workers,             # Number of subprocesses to use for data loading.
    seed=42,                                             # Random seed for reproducibility.
    overwrite_output_dir=True,                           # Overwrite the content of the output directory.
    prediction_loss_only=True,                           # Compute only the prediction loss (ignoring other losses like language modeling loss).
    learning_rate=args.lr,                               # Learning rate for the optimizer.
    adam_beta1=0.9,                                      # Beta1 hyperparameter for the Adam optimizer.
    adam_beta2=0.98,                                     # Beta2 hyperparameter for the Adam optimizer.
    max_grad_norm=args.clip_norm,                        # Maximum gradient norm for gradient clipping.
    warmup_steps=args.warmup_steps,                      # Number of warmup steps for the learning rate scheduler.
    lr_scheduler_type="linear",                    # Type of learning rate scheduler.
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    resume_from_checkpoint=True,
    )                           

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset)

    trainer.train()

if __name__ == "__main__":
    main()