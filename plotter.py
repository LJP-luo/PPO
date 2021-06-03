import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np

DIV_LINE_WIDTH = 50


def plot_data(data, xaxis='Epoch', value="AverageEpRet", smooth=1, **kwargs):
    if smooth > 1:
        """
        smooth data with moving window average.
        that is,
            smoothed_y[t] = average(y[t-k], y[t-k+1], ..., y[t+k-1], y[t+k])
        where the "smooth" param is width of that window (2k+1)
        """
        y = np.ones(smooth)
        for datum in data:
            x = np.asarray(datum[value])
            z = np.ones(len(x))
            smoothed_x = np.convolve(x, y, 'same') / np.convolve(z, y, 'same')
            datum[value] = smoothed_x

    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)
    sns.set(style="darkgrid", font_scale=1.)
    sns.relplot(data=data, x=xaxis, y=value, hue='Algorithm', ci='sd', kind='line', **kwargs)

    # plt.legend(loc='best').set_draggable(True)
    # plt.legend(loc='upper center', ncol=3, handlelength=1,
    #           borderaxespad=0., prop={'size': 13})

    xscale = np.max(np.asarray(data[xaxis])) > 5e3
    if xscale:
        # Just some formatting niceness: x-axis scale in scientific notation if max x is large
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    # plt.tight_layout(pad=0.5)


def get_datasets(logdir):
    """
    Recursively look through logdir for output files produced by
    spinup.logx.Logger.
    Assumes that any file "progress.txt" is a valid hit.
    """
    datasets = []
    for root, _, files in os.walk(logdir):
        if 'progress.txt' in files:
            exp_name = None
            try:
                config_path = open(os.path.join(root, 'config.json'))
                config = json.load(config_path)
                if 'exp_name' in config:
                    exp_name = config['exp_name']
            except:
                print('No file named config.json')

            try:
                exp_data = pd.read_table(os.path.join(root, 'progress.txt'))
            except:
                print('Could not read from %s' % os.path.join(root, 'progress.txt'))
                continue
            # Performance = 'AverageTestEpRet' if 'AverageTestEpRet' in exp_data else 'AverageEpRet'
            # Performance = 'AverageEpRet'
            # exp_data.insert(len(exp_data.columns), 'Performance', exp_data[Performance])
            exp_data.insert(len(exp_data.columns), 'Algorithm', exp_name)
            datasets.append(exp_data)
    return datasets


def get_all_datasets(all_logdirs):

    logdirs = []
    for logdir in all_logdirs:
        if osp.isdir(logdir):
            logdirs += [logdir]

    # Verify logdirs
    print('Plotting from...\n' + '=' * DIV_LINE_WIDTH + '\n')
    for logdir in logdirs:
        print(logdir)
    print('\n' + '=' * DIV_LINE_WIDTH)

    # Make sure the legend is compatible with the logdirs
    # assert not legend or (len(legend) == len(logdirs)), \
    #     "Must give a legend title for each set of experiments."

    # Load data from logdirs
    data = []
    # if legend:
    #     for log, leg in zip(logdirs, legend):
    #         data += get_datasets(log, leg)
    # else:
    #     for log in logdirs:
    #         data += get_datasets(log)
    for logdir in logdirs:
        data += get_datasets(logdir)

    return data


def make_plots(all_logdirs, xaxis='TotalEnvInteracts', value='AverageEpRet', smooth=1, estimator='mean'):

    data = get_all_datasets(all_logdirs)

    estimator = getattr(np, estimator)  # choose what to show on main curve: mean? max? min?
    plt.figure()
    plot_data(data=data, xaxis=xaxis, value=value, smooth=smooth, estimator=estimator)
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*', default='',
                        help='E.g. [experiments/PPO_HalfCCheetah-v2/seed_x, ...]')
    parser.add_argument('--xaxis', '-x', default='Time')
    parser.add_argument('--value', '-y', default='AverageEpRet')
    parser.add_argument('--smooth', '-s', type=int, default=1)
    parser.add_argument('--est', default='mean')
    args = parser.parse_args()

    make_plots(all_logdirs=args.logdir, xaxis=args.xaxis, value=args.value,
               smooth=args.smooth, estimator=args.est)


if __name__ == "__main__":
    main()
