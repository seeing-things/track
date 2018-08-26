"""Laser pointer control.

Defines a single class LaserPointer that allows control of a laser pointer via an FTDI device. This
module assumes that an electrical interface between a single pin of the FTDI device and the laser
pointer allow the laser pointer on/off state to be controlled by the voltage level on that pin.
The details of that electrical interface are not specified.
"""

from pyftdi.ftdi import Ftdi


# for purposes of set_bitmode bitmask:
# bit   set => output
# bit unset =>  input
PIN_TXD = (1<<0)
PIN_RXD = (1<<1)
PIN_RTS = (1<<2)
PIN_CTS = (1<<3)
PIN_DTR = (1<<4)
PIN_DSR = (1<<5)
PIN_DCD = (1<<6)
PIN_RI  = (1<<7)


class LaserPointer(object):
    """Class for controlling a laser pointer via an FTDI device.

    This class allows a laser pointer to be controlled by toggling the state of a pin on an FTDI
    USB Serial device in bitbang mode.

    Attributes:
        laser_pin: An integer bitmask with a single bit set corresponding to the pin that the laser
            pointer is controlled by. See the set of constants defined at the top of this source
            file.
        ftdi: An object of type pyftdi.ftdi.Ftdi.
    """

    def __init__(self, ftdi_pid='232r', serial_num=None, laser_pin=PIN_CTS):
        """Inits LaserPointer object.

        Initializes a LaserPointer object by constructing an Ftdi object and opening the desired
        FTDI device. Configures the FTDI device to bitbang mode.

        Args:
            ftdi_pid: Product ID string for the FTDI device.
            serial_num: String containing the serial number of the FTDI device. If None, any
                serial number will be accepted.
            laser_pin: Integer bit mask with a single bit set corresponding to the pin on the FTDI
                device that controls the laser pointer.
        """
        self.laser_pin = laser_pin

        self.ftdi = Ftdi()

        self.ftdi.open_bitbang(
            vendor=Ftdi.VENDOR_IDS['ftdi'],
            product=Ftdi.PRODUCT_IDS[Ftdi.FTDI_VENDOR][ftdi_pid],
            serial=serial_num,        # serial number of FT232RQ in the laser FTDI cable
            latency=Ftdi.LATENCY_MAX, # 255ms; reduce if necessary (at cost of higher CPU usage)
        )

        # set laser_pin as output, all others as inputs
        self.ftdi.set_bitmode(self.laser_pin, Ftdi.BITMODE_BITBANG)
        assert self.ftdi.bitbang_enabled

    def __del__(self):
        self.ftdi.close()

    def set(self, enable):
        """Sets the state of the laser pointer.

        Args:
            enable: Laser pointer will be turned on when True and off when False.
        """
        if enable:
            self.ftdi.write_data(bytes([0x00]))
        else:
            self.ftdi.write_data(bytes([self.laser_pin]))

    def get(self):
        """Gets the current state of the laser pointer.

        This reads the current state from FTDI device and does not rely on any state information
        in this class.

        Returns:
            The current state of the laser pointer as a boolean.
        """
        return (self.ftdi.read_pins() & self.laser_pin) == 0x0
