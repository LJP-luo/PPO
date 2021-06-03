"""
Some simple logging functionality, inspired by rllab's logging.
Logs to a tab-separated-values file (path/to/output_directory/progress.txt)
"""
import json
import numpy as np
import torch
import os
import atexit

from utils.mpi_tools import process_id, mpi_statistics_scalar
from utils.serialization import convert_json

color2num = dict(
    gray=30,
    red=31,
    green=32,
    yellow=33,
    blue=34,
    magenta=35,
    cyan=36,
    white=37,
    crimson=38
)


def colorize(string, color, bold=False, highlight=False):
    """
    Colorize a string.
    This function was originally written by John Schulman.
    """
    attr = []
    num = color2num[color]
    if highlight:
        num += 10
    attr.append(str(num))
    if bold:
        attr.append('1')
    return '\x1b[%sm%s\x1b[0m' % (';'.join(attr), string)


class Logger:
    """
    A general-purpose logger.
    Makes it easy to save diagnostics, hyper-parameter configurations,
    the state of a training run, and the trained model.
    """

    def __init__(self, output_dir=None, output_fname='progress.txt',
                 exp_name=None, env_name=None, seed=0):

        if process_id() == 0:
            self.output_dir = output_dir or f"{os.path.abspath('.')}" \
                                            f"/experiments/{exp_name}_{env_name}/seed_{seed}"
            if os.path.exists(self.output_dir):
                print(f"Warning: Log dir {self.output_dir} "
                      f"already exists! Storing info there anyway.")
            else:
                os.makedirs(self.output_dir)
            self.output_file = open(os.path.join(self.output_dir, output_fname), 'w')
            atexit.register(self.output_file.close)
            print(colorize(f"Logging data to {self.output_file.name}", 'green', bold=True))
        else:
            self.output_dir = None
            self.output_file = None
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name

    @staticmethod
    def log_msg(msg, color='green'):
        """Print a colorized message to stdout."""
        if process_id() == 0:
            print(colorize(msg, color, bold=True))

    def log_tabular(self, key, val):
        """
        Log a value of some diagnostic.
        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, \
                f"Trying to introduce a new key {key} that you didn't include in the first iteration"
        assert key not in self.log_current_row,\
            f"You already set {key} this iteration. Maybe you forgot to call dump_tabular()"
        self.log_current_row[key] = val

    def save_config(self, config):
        """
        Log an experiment configuration.
        Call this once at the top of your experiment, passing in all important
        config vars as a dict. This will serialize the config to JSON, while
        handling anything which can't be serialized in a graceful way (writing
        as informative a string as possible).
        Example use:
        .. code-block:: python
            logger = EpochLogger(**logger_kwargs)
            logger.save_config(locals())
        """
        config_json = convert_json(config)
        if self.exp_name is not None:
            config_json['exp_name'] = self.exp_name
        if process_id() == 0:
            output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
            print(colorize('Saving config:\n', color='cyan', bold=True))
            print(output)
            with open(os.path.join(self.output_dir, "config.json"), 'w') as out:
                out.write(output)

    def save_policy(self, state_dict):
        """
        Saves the state of an experiment.
        To be clear: this is about saving *state*, not logging diagnostics.
        All diagnostic logging is separate from this function.
        If you only want to maintain a single state and overwrite it at each
        call with the most recent version, leave ``itr=None``.
        If you want to keep all of the states you save, provide unique (increasing) values for 'itr'.
        Args:
            state_dict (dict): Dictionary containing essential elements to
                  describe the current state of training.
            itr: An int, or None. Current iteration of training.
        """
        if process_id() == 0:
            # fname = 'vars.pth' if itr is None else f'vars{itr}.pth'
            fpath = os.path.join(self.output_dir, 'model')
            if not os.path.exists(fpath):
                os.makedirs(fpath)
            fname = f'{self.exp_name}-actor_critic.pth'
            fname = os.path.join(fpath, fname)

            try:
                torch.save(state_dict, fname)
            except:
                self.log_msg('Warning: could not save state_dict.', color='red')

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.
        Writes both to stdout, and to the output file.
        """
        if process_id() == 0:
            vals = []
            key_lens = [len(key) for key in self.log_headers]
            max_key_len = max(15, max(key_lens))
            keystr = '%' + '%d' % max_key_len
            fmt = "| " + keystr + "s | %15s |"
            n_slashes = 22 + max_key_len

            # print current epoch diagnostics
            print("-" * n_slashes)
            for key in self.log_headers:
                val = self.log_current_row.get(key, "")
                valstr = "%8.3g" % val if hasattr(val, "__float__") else val
                print(fmt % (key, valstr))
                vals.append(val)
            print("-" * n_slashes, flush=True)

            # write epoch diagnostic to output_file (./experiments/exp_name_time/progress.txt)
            if self.output_file is not None:
                if self.first_row:
                    self.output_file.write("\t".join(self.log_headers) + "\n")
                self.output_file.write("\t".join(map(str, vals)) + "\n")
                self.output_file.flush()
        self.log_current_row.clear()
        self.first_row = False


class EpochLogger(Logger):
    """
    A variant of Logger tailored for tracking average values over epochs.
    With an EpochLogger, each time the quantity is calculated,
    you would use
    .. code-block:: python
        epoch_logger.store(NameOfQuantity=quantity_value)

    to load it into the EpochLogger's state. Then at the end of the epoch,
    you would use
    .. code-block:: python
        epoch_logger.log_tabular(NameOfQuantity, **options)
    to record the desired values.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_dict = dict()

    def store(self, **kwargs):
        """
        Save something into the epoch_logger's current state.
        Provide an arbitrary number of keyword arguments with numerical
        values.
        """
        for k, v in kwargs.items():
            if not (k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.
        Args:
            key (string): The name of the diagnostic.
            val: A value for the diagnostic.
            with_min_and_max (bool): If true, log min and max values of the
                diagnostic over the epoch.
            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            super().log_tabular(key, val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
            stats = mpi_statistics_scalar(vals, with_min_and_max=with_min_and_max)
            super().log_tabular(key if average_only else 'Average' + key, stats[0])
            if not average_only:
                super().log_tabular('Std' + key, stats[1])
            if with_min_and_max:
                super().log_tabular('Max' + key, stats[3])
                super().log_tabular('Min' + key, stats[2])
        self.epoch_dict[key] = []

    def get_stats(self, key):
        """
        Lets an algorithm ask the logger for mean/std/min/max of a diagnostic.
        """
        v = self.epoch_dict[key]
        vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape) > 0 else v
        return mpi_statistics_scalar(vals)





