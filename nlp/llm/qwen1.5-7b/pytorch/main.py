import sys

from firefly.train import init_components, setup_everything
from loguru import logger


def train():
    # 进行一些配置和检查
    args, training_args = setup_everything()
    # 加载各种组件
    trainer = init_components(args, training_args)
    # 开始训练
    logger.info("*** starting training ***")
    # todo resume from checkpoint
    # https://github.com/huggingface/transformers/issues/24252
    train_result = trainer.train()
    # 保存最后的checkpoint
    trainer.save_model(training_args.output_dir)  # Saves the tokenizer too
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == "__main__":
    train()
