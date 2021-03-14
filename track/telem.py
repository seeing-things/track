"""telemetry logging to a time series database"""

from abc import ABC, abstractmethod
import os
from datetime import datetime
import pathlib
import time
import threading
import toml
import traceback
from typing import Dict, List, Optional
from configargparse import Namespace
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import WriteOptions
from influxdb_client.client.exceptions import InfluxDBError
from track.config import ArgParser


class TelemSource(ABC):
    """Abstract base class for a producer of telemetry.

    All classes which produce telemetry information to be logged to the time series database must
    inherit from this class.
    """

    @abstractmethod
    def get_telem_points(self) -> List[Point]:
        """Get telemetry points.

        Gets zero or more telemetry points from this object for the purpose of writing them to the
        database. Any points returned in a call to this method should not be returned again in a
        subsequent call.

        Returns:
            A list of zero or more Point objects.
        """


class TelemLogger:
    """Logs telemetry to a time-series database.

    This class creates a connection to an InfluxDB database, samples telemetry
    from various TelemSource objects, and writes this data to the database.
    Telemetry is sampled in a thread that executes at a regular rate.

    Attributes:
        bucket: Name of bucket in the database to write to.
        influxdb_client: InfluxDBClient object.
        period: Telemetry period in seconds.
        running: Boolean indicating whether the logger is running or not.
        sources: Dict of TelemSource objects to be polled for telemetry.
        thread: Thread for sampling telemetry from sources.
        write_api: InfluxDBClient write API.
    """

    def __init__(
        self,
        influx_config_filename: str,
        bucket: str = 'telem',
        period: Optional[float] = None,
        sources: Optional[Dict[str, TelemSource]] = None,
    ):
        """Inits a TelemLogger object.

        Establishes a connection with the InfluxDB database and creates the worker thread.
        Telemetry sampling will not occur until the start() method is called.

        Args:
            influx_config_filename: Filename of the InfluxDB CLI configuration file, typically
                $HOME/.influxdbv2/configs which is created by running `influx setup`.
            bucket: Name of bucket in database to write to.
            period: Telemetry will be sampled asynchronously at this interval in seconds if set to
                a positive value. If None this object will only poll sources for telemetry when
                `poll_sources()` is called.
            sources: Dict of TelemSource objects to be polled for telemetry. Values should be
                objects of type TelemSource. Keys are only used to prevent registering the same
                object more than once.
        """
        self.bucket = bucket
        self.period = period

        print(f'period: {self.period} s')

        if sources is not None:
            self.sources = sources
        else:
            self.sources = {}

        # Can't use `InfluxDBClient.from_config_file()` here because `influx setup` generates a
        # TOML formatted config file, but the Python client expects a .ini file. Because of course.
        config_file_dict = toml.load(influx_config_filename)
        self.influxdb_client = InfluxDBClient(
            url=config_file_dict['influx2']['url'],
            token=config_file_dict['influx2']['token'],
            org=config_file_dict['influx2']['org'],
        )

        self.write_api = self.influxdb_client.write_api(
            write_options=WriteOptions(
                batch_size=500,  # number of data points per batch
                flush_interval=1000,  # milliseconds before batch is written
                jitter_interval=0,
                retry_interval=1000,  # milliseconds before retry after failed write
                max_retries=5,
                max_retry_delay=10_000,  # give up after this many milliseconds
                exponential_base=2
            )
        )

        if self.period is not None:
            self.thread = threading.Thread(
                target=self._worker_thread,
                name='TelemLogger: worker thread'
            )
            self.running = False

    def start(self) -> None:
        """Start sampling telemetry asynchronously.

        Raises:
            RuntimeError if no period was defined.
        """
        if self.period is not None:
            self.running = True
            self.thread.start()
        else:
            raise RuntimeError('No period was defined')

    def stop(self) -> None:
        """Stop sampling telemetry asynchronously.

        Raises:
            RuntimeError if no period was defined.
        """
        if self.period is not None:
            self.running = False
        else:
            raise RuntimeError('No period was defined')

    def register_sources(self, sources: Dict[str, TelemSource]) -> None:
        """Register one or more telemetry source object such that it is polled by this logger.

        Note that this adds to any sources already registered. Existing sources are not removed
        unless there are key collisions.

        Args:
            sources: Dict of TelemSource objects to be polled for telemetry. Values should be
                objects of type TelemSource. Keys are only used to prevent registering the same
                object more than once.
        """
        self.sources.update(sources)

    def poll_sources(self) -> None:
        """Poll all registered `TelemSource` objects and post to the database"""
        for source in self.sources.values():
            for point in source.get_telem_points():
                # tag point with the class name of the object that generated it
                point.tag('class', type(source).__name__)
                self.write_api.write(bucket=self.bucket, record=point)

    def post_points(self, points: List[Point]) -> None:
        """Write points to the database.

        Args:
            points: A list of zero or more Point objects to be written to the database.
        """
        self.write_api.write(bucket=self.bucket, record=points)

    def _worker_thread(self) -> None:
        """Gathers telemetry and posts to database once per sample period."""

        # Make sure this thread does not have realtime priority
        os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))

        while True:
            if not self.running:
                return
            start_time = time.time()
            try:
                self.poll_sources()
            except InfluxDBError as e:
                print('Failed to post telemetry to database: ' + str(e))
                traceback.print_exc()
            elapsed_time = time.time() - start_time
            sleep_time = self.period - elapsed_time
            if sleep_time > 0.0:
                time.sleep(sleep_time)


def add_program_arguments(parser: ArgParser, synchronous: bool = False) -> None:
    """Add program arguments relevant to telemetry.

    Args:
        parser: The instance of ArgParser to which this function will add arguments.
        synchronous: If True, the assumption is that telemetry will only be polled synchronously.
            In this case the telem-period program argument is omitted.
    """
    telem_group = parser.add_argument_group(
        title='Telemetry Logger Options',
        description='Options that apply to telemetry logging',
    )
    telem_group.add_argument(
        '--telem-enable',
        help='enable logging of telemetry to database',
        action='store_true'
    )
    telem_group.add_argument(
        '--telem-influxdb-configfile',
        help='filename of InfluxDB CLI config file',
        default=str(pathlib.Path.home().joinpath('.influxdbv2/configs'))
    )
    if not synchronous:
        telem_group.add_argument(
            '--telem-period',
            help='telemetry sampling period in seconds',
            default=1.0,
            type=float
        )


def make_telem_logger_from_args(
        args: Namespace,
        sources: Optional[dict] = None
    ) -> Optional[TelemLogger]:
    """Construct a TelemLogger based on the program arguments provided.

    Args:
        args: Set of program arguments.
        sources: Dict of TelemSources to be passed to TelemLogger constructor.

    Returns:
        An instance of TelemLogger if the `--telem-enable` flag is set, otherwise returns None.
    """
    if args.telem_enable:
        return TelemLogger(
            influx_config_filename=args.telem_influxdb_configfile,
            period=(args.telem_period if 'telem_period' in args else None),
            sources=sources,
        )
    else:
        return None
