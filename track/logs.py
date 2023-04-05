"""Configure logging for the package."""

from enum import IntEnum
from importlib import metadata
import logging
from logging.handlers import RotatingFileHandler
import os
import subprocess
import sys
import time
import appdirs
import coloredlogs
from configargparse import Namespace
from track.config import ArgParser


LOG_PATH = appdirs.user_log_dir(__package__)

logger = logging.getLogger(__name__)


class LogLevel(IntEnum):
    """Define an enum for common levels since the logging module doesn't."""

    CRITICAL = logging.CRITICAL
    ERROR = logging.ERROR
    WARNING = logging.WARNING
    INFO = logging.INFO
    DEBUG = logging.DEBUG


def get_git_commit_hash() -> str | None:
    """Returns the git commit hash if a git repository.

    Returns:
        The git commit hash as a hex string, or None if not a git repository. If a git repository
        and there are locally modified or untracked files, the word "dirty" will be appended after
        the commit hash.
    """
    try:
        commit_hash = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(os.path.realpath(__file__)),
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return None
    try:
        git_status = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True,
            check=True,
            cwd=os.path.dirname(os.path.realpath(__file__)),
        ).stdout.strip()
    except subprocess.CalledProcessError:
        return None

    if len(git_status) > 0:
        return f'{commit_hash} (dirty)'
    return commit_hash


def setup(
    program_name: str,
    console_level: int,
    file_level: int,
    enable_console_logging: bool,
) -> None:
    """Setup logging for the package.

    This sets up logging to both the console and a log file. This should ideally be called before
    any logging is performed in other modules.

    To log from other modules, get a Logger object by calling `logging.getLogger(__name__)`.

    Args:
        program_name: Name of the program. The log file will be the program name with .log appended.
        console_level: Log level for the console handler.
        file_level: Log level for the file handler.
        enable_console_logging: Enable logging to the console (stdout).
    """
    log_filename = os.path.join(LOG_PATH, f'{program_name}.log')

    root_logger = logging.getLogger(__package__)
    root_logger.setLevel(min(file_level, console_level))

    formatter = logging.Formatter(
        fmt='[%(asctime)s.%(msecs)d] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    formatter.converter = time.gmtime  # use UTC rather than local time

    os.makedirs(LOG_PATH, exist_ok=True)
    file_handler = RotatingFileHandler(
        filename=log_filename,
        maxBytes=10_000_000,  # 10 MB
        backupCount=10,
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    if enable_console_logging:
        formatter = coloredlogs.ColoredFormatter(
            fmt='[%(asctime)s.%(msecs)d] [%(name)s] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            level_styles={
                'critical': {'color': 'red', 'bold': True},
                'error': {'color': 'red', 'bright': True},
                'warning': {'color': 'yellow', 'bright': True},
                'info': {'color': 'white'},
                'debug': {'color': 'white', 'faint': True},
            },
            field_styles={
                'asctime': {'color': 'green'},
                'name': {'color': 'blue', 'bright': True},
                'levelname': {'color': 'magenta'},
            },
        )
        formatter.converter = time.gmtime  # use UTC rather than local time

        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Called when an exception is unhandled. The default hook prints the traceback to stdout.
    sys.excepthook = lambda *args: logger.critical('Unhandled exception.', exc_info=args)

    # Using critical for version info so it always gets logged no matter the level
    logger.critical(f'Program started with the following arguments: {sys.argv}')
    logger.critical(f'{__package__} package version {metadata.version(__package__)}')
    if (commit_hash := get_git_commit_hash()) is not None:
        logger.critical(f'Git commit: {commit_hash}')
    logger.debug(f'PID: {os.getpid()}')
    logger.info(f'Console log level set to {LogLevel(console_level).name}')
    logger.info(f'File log level set to {LogLevel(file_level).name}')
    logger.info(f'Log file: {log_filename}')


def add_program_arguments(parser: ArgParser) -> None:
    """Add program arguments for logging.

    Args:
        parser: The instance of ArgParser to which this function will add arguments.
    """
    logging_group = parser.add_argument_group(
        title='Logging Options',
        description='Options that apply to logging',
    )
    logging_group.add_argument(
        '--console-log-level',
        help='log level for console',
        choices=tuple(level.name for level in LogLevel),
        default=LogLevel.INFO.name,
    )
    logging_group.add_argument(
        '--file-log-level',
        help='log level for log file',
        choices=tuple(level.name for level in LogLevel),
        default=LogLevel.WARNING.name,
    )
    logging_group.add_argument(
        '--no-console-logs',
        help='disable logging to the console',
        action='store_false',
        dest='enable_console_logging',
    )


def setup_logging_from_args(args: Namespace, program_name: str) -> None:
    """Setup logging using program arguments."""
    setup(
        program_name,
        console_level=LogLevel[args.console_log_level.upper()],
        file_level=LogLevel[args.file_log_level.upper()],
        enable_console_logging=args.enable_console_logging,
    )
