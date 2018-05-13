from abc import ABC, abstractmethod
import time
import threading
import influxdb
import datetime

class TelemSource(ABC):
    """Abstract base class for a producer of telemetry.

    All classes which produce telemetry information to be logged to the time
    series database must inherit from this class.
    """

    @abstractmethod
    def get_telem_channels(self):
        """Get telemetry channels.

        Gets telemetry information from the object. Zero or more channels of
        telemetry are produced. For each channel a single value is obtained
        which represents the state of that channel at the moment the function
        is called.

        Returns:
            A dict with telemetry data. Keys are channel names.
        """
        pass


class TelemLogger(object):
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
        host='localhost',
        port=8086,
        dbname='telem',
        period=1.0,
        sources={}
    ):
        """Inits a TelemLogger object.

        Establishes a connection with the InfluxDB database and creates the
        worker thread. Telemetry sampling will not occur until the start()
        method is called.

        Args:
            host: Hostname of machine running the InfluxDB database.
            port: Port number the InfluxDB database listens on.
            dbname: Name of database.
            period: Telemetry will be sampled at this interval in seconds.
            sources: Dict of TelemSource objects to be polled for telemetry.
                Keys will be used as the measurement names in the database.
                Values should be objects of type TelemSource.
        """
        self.db = influxdb.InfluxDBClient(host=host, port=port, database=dbname)
        self.thread = threading.Thread(target=self._worker_thread)
        self.period = period
        self.sources = sources
        self.running = False

    def start(self):
        """Start sampling telemetry."""
        self.running = True
        self.thread.start()

    def stop(self):
        """Stop sampling telemetry."""
        self.running = False

    def _post_point(self, name, channels):
        """Write a sample of telemetry channels to the database.

        Writes one sample from one or more telemetry channels to the database.
        A single timestamp generated from the current system time is applied to
        all channels. The channels are added as fields of a single point in the
        named measurement.

        Args:
            name: Name of this collection of channels. This corresponds to the
                measurement name in the InfluxDB database.
            channels: A dict containing the telemetry samples. The keys give
                the channel names, which become field names in the InfluxDB
                measurement.
        """
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        json_body = [
            {
                "measurement": name,
                "time": timestamp,
                "fields": channels
            }
        ]
        self.db.write_points(json_body)

    def _worker_thread(self):
        """Gathers telemetry and posts to database once per sample period."""
        while True:
            if not self.running:
                return
            start_time = time.time()
            try:
                for name, source in self.sources.items():
                    self._post_point(name, source.get_telem_channels())
            except Exception as e:
                print('Failed to post telemetry to database, logger shutting down: ' + str(e))
                self.running = False
                return
            elapsed_time = time.time() - start_time
            sleep_time = self.period - elapsed_time
            if sleep_time > 0.0:
                time.sleep(sleep_time)