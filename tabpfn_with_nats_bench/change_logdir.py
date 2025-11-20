import argparse
import os
import datetime
import regex as re
import __main__


def change_logdir_if_default(main_args: argparse.Namespace):
    """Change logdir if it is the default value."""
    # check if logdir is default
    if main_args.logdir == 'logs':
        # Create logdir name.
        main_args.logdir = os.path.join("logs", "{}-{}-{}".format(
            os.path.basename(__main__.__file__),
            datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
            ",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", k), v) for k, v in sorted(vars(main_args).items())))
        ))
    os.makedirs(main_args.logdir, exist_ok=True)


