"""Registers a handler for certain signals to allow for graceful shutdown."""

import logging
import signal
import sys


logger = logging.getLogger(__name__)


# List of signals to handle with `shutdown_handler`
SHUTDOWN_SIGNALS = [signal.SIGHUP, signal.SIGINT, signal.SIGQUIT, signal.SIGTERM]


# pylint: disable=unused-argument
def shutdown_handler(signum: int, frame) -> None:
    """Custom handler for some common shutdown signals.

    The objective here is to ensure that the `SystemExit` exception is raised once and only once at
    shutdown, such that cleanup code (`__exit__` methods, for example) runs to completion. Thus
    this handler masks future signals so that no additional exceptions are raised.

    An unavoidable race condition exists; if this handler is running and another shutdown signal
    arrives before it has been masked then this handler could be executed multiple times resulting
    in multiple `SystemExit` exceptions being raised. In practice this is (hopefully) unlikely.
    """
    # pylint: disable=no-member
    logger.info(f'Handling {signal.Signals(signum).name} by raising `SystemExit`')
    for sig in SHUTDOWN_SIGNALS:
        signal.signal(sig, signal.SIG_IGN)  # ignore from now on
    sys.exit(0)  # raises the SystemExit exception


def register_handlers() -> None:
    """Register handlers for these signals."""
    for sig in SHUTDOWN_SIGNALS:
        signal.signal(sig, shutdown_handler)

register_handlers()
