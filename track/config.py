"""Convenience class inherited from configargparse.ArgParser with project-special defaults."""

import configargparse


DEFAULT_CONFIG_FILES = [
    './track.cfg',
    '~/.track.cfg',
]


class ArgumentDefaultsHelpFormatterImproved(configargparse.ArgumentDefaultsHelpFormatter):
    """Same as ArgumentDefaultsHelpFormatter, but won't print "(default: None)" nonsense.

    Technically, "only the name of [the ArgumentsDefaultHelpFormatter] class is considered a
    public API", but we don't have to follow their dumb rules.
    """

    def _get_help_string(self, action):
        if action.default is None:
            return action.help
        else:
            return super()._get_help_string(action)


class ArgParser(configargparse.ArgParser):
    """Uses the contructor arguments we care about in this project.
    """

    def __init__(self, **kwargs):
        super(ArgParser, self).__init__(
            ignore_unknown_config_file_keys=True,
            allow_abbrev=True,
            default_config_files=DEFAULT_CONFIG_FILES,
            formatter_class=ArgumentDefaultsHelpFormatterImproved,
            config_file_parser_class=configargparse.DefaultConfigFileParser, # INI format
            args_for_setting_config_path=['-c', '--cfg'],
            args_for_writing_out_config_file=['-w', '--cfg-write'],
            **kwargs
        )
