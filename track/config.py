"""Convenience class inherited from configargparse.ArgParser with project-special defaults."""

import configargparse


DEFAULT_CONFIG_FILES = [
    './track.cfg',
    '~/.track.cfg',
]


class ArgParser(configargparse.ArgParser):
    """Uses the contructor arguments we care about in this project.
    """

    def __init__(self, **kwargs):
        super(ArgParser, self).__init__(
            ignore_unknown_config_file_keys=True,
            allow_abbrev=True,
            default_config_files=DEFAULT_CONFIG_FILES,
            #formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
            formatter_class=configargparse.RawDescriptionHelpFormatter,
            config_file_parser_class=configargparse.DefaultConfigFileParser, # INI format
            args_for_setting_config_path=['-c', '--cfg'],
            args_for_writing_out_config_file=['-w', '--cfg-write'],
            **kwargs
        )
