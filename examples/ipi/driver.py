# i-PI loads this file (<pes_path> driver.py, or i-pi-py_driver -m custom -P driver.py); the driver is ilmlip.ipi.
from ilmlip.ipi import Driver  # noqa: F401

__DRIVER_CLASS__ = "Driver"
