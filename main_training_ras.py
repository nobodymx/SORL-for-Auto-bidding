import gin
from absl import flags
from absl import app
from run.run_ras import run_ras
import torch
import numpy as np


@gin.configurable
def main(_arg):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    run_ras()


if __name__ == "__main__":
    flags.DEFINE_multi_string(
        'gin_file', ['./config/training_ras.gin'], 'List of paths to the config files.')
    flags.DEFINE_multi_string(
        'gin_param', None, 'Newline separated list of Gin parameter bindings.')
    FLAGS = flags.FLAGS
    app.run(main)
