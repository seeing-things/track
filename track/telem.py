from abc import ABC, abstractmethod
import os
import time
import threading
import traceback
import influxdb
from influxdb.exceptions import InfluxDBClientError, InfluxDBServerError
import datetime
from configargparse import Namespace
from track.config import ArgParser

class TelemSource(ABC):
    """Abstract base class for a producer of telemetry.

    All classes which produce telemetry information to be logged to the time series database must
    inherit from this class.
    """

    @abstractmethod
    def get_telem_channels(self) -> dict:
        """Get telemetry channels.

        Gets telemetry information from the object. Zero or more channels of telemetry are
        produced. For each channel a single value is obtained which represents the state of that
        channel at the moment the function is called.

        Returns:
            A dict with telemetry data. Keys are channel names.
        """


class TelemLogger:
    """Logs telemetry to a time-series database.

    This class creates a connection to an InfluxDB database, samples telemetry
    from various TelemSource objects, and writes this data to the database.
    Telemetry is sampled in a thread that executes at a regular rate.

    Attributes:
        db: InfluxDBClient object.
        thread: Thread for sampling telemetry from sources.
        period: Telemetry period in seconds.
        sources: Dict of TelemSource objects to be polled for telemetry.
        running: Boolean indicating whether the logger is running or not.
    """

    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8086,
        dbname: str = 'telem',
        period: float = 1.0,
        sources: dict = {}
    ):
        """Inits a TelemLogger object.

        Establishes a connection with the InfluxDB database and creates the worker thread.
        Telemetry sampling will not occur until the start() method is called.

        Args:
            host: Hostname of machine running the InfluxDB database.
            port: Port number the InfluxDB database listens on.
            dbname: Name of database.
            period: Telemetry will be sampled at this interval in seconds.
            sources: Dict of TelemSource objects to be polled for telemetry. Keys will be used as
                the measurement names in the database. Values should be objects of type
                TelemSource.
        """
        self.db = influxdb.InfluxDBClient(host=host, port=port, database=dbname)
        self.thread = threading.Thread(target=self._worker_thread, name='TelemLogger: worker thread')
        self.period = period
        self.sources = sources
        self.running = False

    def start(self) -> None:
        """Start sampling telemetry."""
        self.running = True
        self.thread.start()

    def stop(self) -> None:
        """Stop sampling telemetry."""
        self.running = False

    def _post_point(self, name: str, channels: dict) -> None:
        """Write a sample of telemetry channels to the database.

        Writes one sample from one or more telemetry channels to the database. A single timestamp
        generated from the current system time is applied to all channels. The channels are added
        as fields of a single point in the named measurement.

        Args:
            name: Name of this collection of channels. This corresponds to the measurement name in
                the InfluxDB database.
            channels: A dict containing the telemetry samples. The keys give the channel names,
                which become field names in the InfluxDB measurement. If empty nothing is written
                to the database.
        """
        if not channels:
            return
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        json_body = [
            {
                "measurement": name,
                "time": timestamp,
                "fields": channels
            }
        ]
        self.db.write_points(json_body)

    def _worker_thread(self) -> None:
        """Gathers telemetry and posts to database once per sample period."""

        # Make sure this thread does not have realtime priority
        os.sched_setscheduler(0, os.SCHED_OTHER, os.sched_param(0))

        while True:
            if not self.running:
                return
            start_time = time.time()
            try:
                for name, source in self.sources.items():
                    self._post_point(name, source.get_telem_channels())
            except (InfluxDBClientError, InfluxDBServerError) as e:
                print('Failed to post telemetry to database: ' + str(e))
                traceback.print_exc()
            elapsed_time = time.time() - start_time
            sleep_time = self.period - elapsed_time
            if sleep_time > 0.0:
                time.sleep(sleep_time)


def add_program_arguments(parser: ArgParser) -> None:
    """Add program arguments relevant to telemetry.

    Args:
        parser: The instance of ArgParser to which this function will add arguments.
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
        '--telem-db-host',
        help='hostname of InfluxDB database server',
        default='localhost'
    )
    telem_group.add_argument(
        '--telem-db-port',
        help='port number of InfluxDB database server',
        default=8086,
        type=int
    )
    telem_group.add_argument(
        '--telem-period',
        help='telemetry sampling period in seconds',
        default=1.0,
        type=float
    )


def make_telem_logger_from_args(args: Namespace, sources: dict = {}) -> TelemLogger:
    """Construct a TelemLogger based on the program arguments provided.

    Args:
        args: Set of program arguments.
        sources: Dict of TelemSources to be passed to TelemLogger constructor.
    """
    return TelemLogger(
        host=args.telem_db_host,
        port=args.telem_db_port,
        period=args.telem_period,
        sources=sources,
    )
