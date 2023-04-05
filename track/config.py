"""Package configuration utilities.

* Defines paths to configuration and data files.
* Specifies the filename of the default shared configuration file used by all programs.
* Provides a convenience child class of configargpars.ArgParser with project-specific defaults.
"""


import logging
import os
import appdirs
import configargparse


logger = logging.getLogger(__name__)


# Where configuration and data files should be stored.
CONFIG_PATH = appdirs.user_config_dir('track')
DATA_PATH = appdirs.user_data_dir('track')


# If the same program argument appears in multiple configuration files, the value in the last
# config file in the list takes precedence. If another configuration file not listed here is
# supplied at the command line with --config-file, it is as if that config file is *added* to the
# end of this list, and does not replace this list. The file shared.cfg is optional. It is not
# provided with this package.
DEFAULT_CONFIG_FILES = [
    os.path.join(CONFIG_PATH, 'shared.cfg'),
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
    """Uses the contructor arguments we care about in this project."""

    def __init__(self, additional_config_files: list[str] | None = None, **kwargs):
        if additional_config_files is not None:
            config_files = DEFAULT_CONFIG_FILES + additional_config_files
        else:
            config_files = DEFAULT_CONFIG_FILES

        super().__init__(
            ignore_unknown_config_file_keys=True,
            allow_abbrev=True,
            default_config_files=config_files,
            formatter_class=ArgumentDefaultsHelpFormatterImproved,
            config_file_parser_class=configargparse.DefaultConfigFileParser,  # INI format
            args_for_setting_config_path=['-c', '--config-file'],
            args_for_writing_out_config_file=['-w', '--write-out-config-file'],
            **kwargs,
        )
