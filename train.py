# coding: utf-8

import os
from pathlib import Path

from trainer import Trainer, TrainerArgs
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

input_path = "/content/hifi-gan-train/arabicttstrain/dataset/wavs/"
output_path = "/content/drive/MyDrive/hifi-gan-train/"
input_filelist = "/content/hifi-gan-train/notebooks/datasets/kareem/wavs.txt"


def main():
    Path(output_path).mkdir(parents=True, exist_ok=True)

    config = HifiganConfig(
        batch_size=64,
        eval_batch_size=16,
        num_loader_workers=os.cpu_count(),
        num_eval_loader_workers=os.cpu_count(),
        run_eval=True,
        test_delay_epochs=5,
        epochs=1000,
        seq_len=8192,
        pad_short=2000,
        use_noise_augment=True,
        eval_split_size=10,
        print_step=25,
        print_eval=False,
        mixed_precision=False,
        lr_gen=1e-4,
        lr_disc=1e-4,
        data_path=input_path,
        output_path=output_path,
    )

    # init audio processor
    ap = AudioProcessor(**config.audio.to_dict())

    # load training samples
    eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

    # init model
    model = GAN(config, ap)

    # init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples
    )
    trainer.fit_with_largest_batch_size(2048)


if __name__ == '__main__':
    main()
