import os

import numpy as np
from gwpy.io import hdf5 as io_hdf5
from gwpy.io import registry as io_registry
from gwpy.io.utils import identify_factory
from gwpy.timeseries import TimeSeriesBase
from gwpy.types.io.hdf5 import write_hdf5_series as gwpy_write_hdf5_series

from ..data import HeterodynedData

# -- read ---------------------------------------------------------------------


def read_ascii_series(input_, array_type=HeterodynedData, **kwargs):
    """
    Read a `Series` from an ASCII text file.

    Parameters
    ----------
    input_: str, file
        The ascii text file to read
    array_type: type
        The desired return type. Defaults to :class:`cwinpy.data.HeterodynedData`.
    """

    data = np.loadtxt(input_, **kwargs)

    # get any comment lines from the file
    commentstrs = list(kwargs.get("comments", ["%", "#"]))  # delimiters
    comments = ""

    if input_.endswith(".gz"):
        import gzip

        openfunc = gzip.open
    else:
        openfunc = open

    with openfunc(input_, "r") as fp:
        for line in fp.readlines():
            firstchar = line.strip()[0]  # remove any proceeding whitespace
            if firstchar in commentstrs:
                # strip the comment delimiter and any leading whitespace
                comments += line.strip(firstchar).strip()

    if data.shape[1] < 2:
        raise IOError("Problem reading in data")

    return array_type(data[:, 1:], times=data[:, 0], comments=comments)


@io_hdf5.with_read_hdf5
def read_hdf5_series(
    source, array_type=HeterodynedData, path="HeterodynedData", **kwargs
):
    """
    Read a `Series` from an HDF5 file.

    Parameters
    ----------
    source: str
        The HDF5 file to be read.
    array_type: type
        The desired return type. Defaults to :class:`cwinpy.data.HeterodynedData`.
    """

    dataset = io_hdf5.find_dataset(source, path=path)
    kwargs = dict(dataset.attrs)

    parfiles = {}
    for par in ["par", "injpar"]:
        if par in kwargs:
            # convert parameter file string to file
            import tempfile

            parfiles[par] = tempfile.mkstemp(suffix=".par")[1]
            with open(parfiles[par], "w") as fp:
                fp.write(kwargs[par])
            kwargs[par] = parfiles[par]

    # check whether injected signal is given
    injdata = kwargs.pop("inj_data", None)

    # make sure certain values are integers
    for key in kwargs:
        if key in ["window", "bbminlength", "bbmaxlength"]:
            if np.isfinite(kwargs[key]):
                kwargs[key] = int(kwargs[key])

    # complex time series data
    data = dataset[()]

    # extract the time stamps
    timespans = kwargs.pop("times", None)
    try:
        dt = kwargs.pop("dt")
    except KeyError:
        dt = kwargs.pop("dx")
    if timespans is not None:
        times = np.array([], dtype=np.float)
        for ts in timespans:
            times = np.concatenate((times, np.arange(ts[0], ts[1] + dt / 2, dt)))
        kwargs["times"] = times
    else:
        kwargs["times"] = np.arange(kwargs["x0"], kwargs["x0"] + len(data) * dt, dt)

    filter_history = kwargs.pop("filter_history", None)

    # extract data variances if contained in the file
    vars = kwargs.pop("vars", None)
    if vars is None:
        array = array_type(data, **kwargs)
    else:
        array = array_type(
            np.column_stack((data.real, data.imag, np.sqrt(vars))), **kwargs
        )

    # re-add noise-free injected signal
    if injdata is not None:
        array._inj_data = TimeSeriesBase(
            injdata, times=array.times, channel=array.channel
        )
        array.injection = True
        array.injpar = parfiles["injpar"]

    # add filter history
    if filter_history is not None:
        array.filter_history = filter_history

    for par in ["par", "injpar"]:
        if par in parfiles:
            # remove temporary parameter file
            os.remove(parfiles[par])

    return array


# -- write --------------------------------------------------------------------


def write_ascii_series(series, output, **kwargs):
    """Write a `Series` to a file in ASCII format
    Parameters
    ----------
    series : :class:`~gwpy.data.Series`
        data series to write
    output : `str`, `file`
        file to write to
    See also
    --------
    numpy.savetxt
        for documentation of keyword arguments
    """

    xarr = series.xindex.value
    yarrr = series.value.real
    yarri = series.value.imag

    stds = None
    if kwargs.pop("includestds", False):
        if hasattr(series, "vars"):
            stds = series.stds

    try:
        comments = series.comments
    except AttributeError:
        comments = ""

    if stds is None:
        return np.savetxt(
            output, np.column_stack((xarr, yarrr, yarri)), header=comments, **kwargs
        )
    else:
        return np.savetxt(
            output,
            np.column_stack((xarr, yarrr, yarri, stds)),
            header=comments,
            **kwargs
        )


def write_hdf5_series(series, output, path="HeterodynedData", **kwargs):
    """Write a `Series` to a file in ASCII format
    Parameters
    ----------
    series : :class:`~gwpy.data.Series`
        data series to write
    output : `str`, `file`
        file to write to
    See also
    --------
    numpy.savetxt
        for documentation of keyword arguments
    """

    # set additional attributes to save
    attrs = kwargs.pop("attrs", {})

    if hasattr(series, "detector"):
        attrs["detector"] = series.detector

    for par in ["par", "injpar"]:
        if hasattr(series, par):
            # hold pulsar parameter data as a string
            attrs[par] = str(series.par)

    # save times as start and end times of contiguous chunks (cannot save
    # full set of times for long datasets due to HDF5 header size
    # restrictions)
    dt = series.dt.value
    if not np.allclose(np.diff(series.times.value), dt * np.ones(len(series) - 1)):
        breaks = np.argwhere(np.diff(series.times.value) != dt)[0].tolist()
        attrs["times"] = []
        breaks = [-1] + breaks + [len(series) - 1]
        for i in range(len(breaks) - 1):
            attrs["times"].append(
                (series.times.value[breaks[i] + 1], series.times.value[breaks[i + 1]])
            )

    # remove metadata slots that can't/shouldn't be written as attributes
    slots = tuple()
    origslots = tuple(series._metadata_slots)
    badslots = ["par", "injpar", "laldetector", "running_median", "vars", "xindex"]

    if kwargs.pop("includestds", False):
        # allow vars to be included
        badslots.remove("vars")

    for slot in series._metadata_slots:
        if slot not in badslots:
            slots += (slot,)
    series._metadata_slots = slots

    outseries = gwpy_write_hdf5_series(series, output, path=path, attrs=attrs, **kwargs)
    series._metadata_slots = origslots

    return outseries


# -- register -----------------------------------------------------------------


def register_ascii_series_io(array_type, format="txt", identify=True, **defaults):
    """
    Register ASCII read/write/identify methods for the given array
    """

    def _read(filepath, **kwargs):
        kwgs = defaults.copy()
        kwgs.update(kwargs)
        if "comments" not in kwgs:
            kwgs.update({"comments": ["%", "#"]})
        return read_ascii_series(filepath, array_type=array_type, **kwgs)

    def _write(series, output, **kwargs):
        kwgs = defaults.copy()
        kwgs.update(kwargs)
        return write_ascii_series(series, output, **kwgs)

    io_registry.register_reader(format, array_type, _read)
    io_registry.register_writer(format, array_type, _write)
    if identify:
        io_registry.register_identifier(format, array_type, identify_factory(format))


def register_hdf_series_io(array_type, format="hdf5", identify=True, **defaults):
    """
    Register HDF5 read/write/identify methods for the given array
    """

    def _read(filepath, **kwargs):
        kwgs = defaults.copy()
        kwgs.update(kwargs)
        if "comments" not in kwgs:
            kwgs.update({"comments": ["%", "#"]})
        return read_hdf5_series(filepath, array_type=array_type, **kwgs)

    def _write(series, output, **kwargs):
        kwgs = defaults.copy()
        kwgs.update(kwargs)
        return write_hdf5_series(series, output, **kwgs)

    io_registry.register_reader(format, array_type, _read)
    io_registry.register_writer(format, array_type, _write)
    if identify:
        io_registry.register_identifier(format, array_type, identify_factory(format))
