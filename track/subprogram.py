"""Manage programs run as subprocesses."""


import logging
import signal
import subprocess


logger = logging.getLogger(__name__)


def terminate_subprocess(process: subprocess.Popen, timeout: float = 2.0) -> None:
    """Terminate a subprocess that is still running.

    If the process is still running, send it SIGINT. If it's still alive after the timeout expires
    send SIGKILL. The process should be very dead by the time this function returns.

    A typical use case is to register this with `atexit.register()` shortly after a process is
    created with subprocess.Popen() to ensure that the parent does not leave a zombie process
    behind even if it ends via an unhandled exception or other atypical code path.

    Args:
        process: The process that should be dead.
        timeout: How long to wait after sending SIGINT before resorting to SIGKILL.
    """
    logger.debug(f'In terminate_subprocess for PID {process.pid}')

    if process.poll() is not None:
        logger.debug(f'PID {process.pid} is already dead.')
        # Already dead; no intervention needed.
        return

    logger.debug(f'Sending SIGINT to subprocess with PID {process.pid}.')
    process.send_signal(signal.SIGINT)
    try:
        retcode = process.wait(timeout)
    except subprocess.TimeoutExpired:
        logger.error(f'Subprocess with PID {process.pid} shutdown timeout; resorting to SIGKILL.')
        process.kill()
        return

    if retcode:
        logger.warning(f'Subprocess with PID {process.pid} exited with code {retcode}.')
    else:
        logger.debug(f'Subprocess with PID {process.pid} exited with code {retcode}.')
