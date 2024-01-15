import transformers
import pytorch_lightning as pl
from transformers import TrainingArguments
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from dataloader import *
from trainpl import *
from metrics import *

import os
import yaml
config = yaml.load(open('./config.yaml', 'r'), Loader = yaml.Loader)
sweep = yaml.load(open('./sweep.yaml', 'r'), Loader = yaml.Loader)


def main():

    #seed 고정
    set_seed(config['seed'])

    wandb_logger = WandbLogger(
        project=config['project_name'], 
        entity='xi-vil',
        config=sweep,
        name=config.run_name)
    
    dataloader = Dataloader(
        config.model_name,
        config.batch_size,
        config.num_workers,
        config.shuffle,
        config.train_path,
        config.dev_path,
        config.test_path,
        config.predict_path,
        config.sep_token
    )

    model = Model(
      config.model_name,
        config.lr,
        config.weight_decay,
        config.loss_fn,
        config.warmup_steps,
        config.total_steps,
    )

    model_path = os.path.join(config.model_dir, config.run_name)


  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
    training_args = TrainingArguments(
    output_dir=config['train']['training_args']['output_dir'],                # output directory
    run_name=config['train']['training_args']['run_name'],
    report_to=config['train']['training_args']['report_to'],
    save_total_limit=config['train']['training_args']['save_total_limit'],    # number of total save model.
    save_steps=config['train']['training_args']['save_steps'],                # model saving step.
    logging_dir=config['train']['training_args']['logging_dir'],                  # directory for storing logs
    logging_steps=config['train']['training_args']['logging_steps'],              # log saving step.
    evaluation_strategy=config['train']['training_args']['evaluation_strategy'],  # evaluation strategy to adopt during training
                                                                                  # `no`: No evaluation during training.
                                                                                  # `steps`: Evaluate every `eval_steps`.
                                                                                  # `epoch`: Evaluate every end of epoch.
    eval_steps = config['train']['training_args']['eval_steps'],                  # evaluation step.
    load_best_model_at_end = config['train']['training_args']['load_best_model_at_end'],

    learning_rate=config['train']['training_args']['learning_rate'],       # learning_rate
    num_train_epochs=config['train']['training_args']['num_train_epochs'],    # total number of training epochs
    per_device_train_batch_size=config['train']['training_args']['per_device_train_batch_size'],  # batch size per device during training
    per_device_eval_batch_size=config['train']['training_args']['per_device_eval_batch_size'],    # batch size for evaluation
    warmup_steps=config['train']['training_args']['warmup_steps'],                # number of warmup steps for learning rate scheduler
    weight_decay=config['train']['training_args']['weight_decay']                 # strength of weight decay

  )
    trainer = pl.Trainer(
        model=Model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_dataset,         # training dataset
        eval_dataset=train_dataset,             # evaluation dataset
        compute_metrics=compute_metrics,             # define metrics function
        logger=WandbLogger,
        callbacks=[
            CustomModelCheckpoint(
                monitor='val_loss',
                mode='min',
                save_last = True,
                save_top_k=1,
                save_weights_only=True,
                filename='{epoch}-{val_loss:.4f}-{val_f1:.4f}'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                mode='min'
            ),
            LearningRateMonitor(logging_interval='step')
        ]

                
    )

    trainer.fit(
        model=Model,
        datamodule=dataloader,
    )