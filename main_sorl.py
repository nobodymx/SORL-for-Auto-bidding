import gin
from absl import flags
from absl import app
from run.run_sorl import run_sorl
import torch
import numpy as np

torch.manual_seed(1)
np.random.seed(1)


@gin.configurable
def main(_arg):
    gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
    run_sorl()


if __name__ == "__main__":
    flags.DEFINE_multi_string(
        'gin_file', ['./config/sorl.gin'], 'List of paths to the config files.')
    flags.DEFINE_multi_string(
        'gin_param', None, 'Newline separated list of Gin parameter bindings.')
    FLAGS = flags.FLAGS
    app.run(main)
