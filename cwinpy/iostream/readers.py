import ast
import os

import numpy as np
from gwpy.io import hdf5 as io_hdf5
from gwpy.io import registry as io_registry
from gwpy.io.utils import identify_factory
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
    source: File object
        The HDF5 file to be read.
    array_type: type
        The desired return type. Defaults to :class:`cwinpy.data.HeterodynedData`.
    """

    dataset = io_hdf5.find_dataset(source, path=path)
    datasettimesegments = io_hdf5.find_dataset(source, path=path + "TimeSegments")

    try:
        datasetstds = io_hdf5.find_dataset(source, path=path + "Sigmas")
    except KeyError:
        datasetstds = None

    kwargs = dict(dataset.attrs)

    # remove any None attributes
    for key in list(kwargs.keys()):
        if str(kwargs[key]) == "None":
            kwargs.pop(key)

    parfiles = {}
    for par in ["par", "injpar"]:
        if par in kwargs:
            # convert parameter file string to file
            import tempfile

            parfiles[par] = tempfile.mkstemp(suffix=".par")[1]
            with open(parfiles[par], "w") as fp:
                fp.write(kwargs[par])
            kwargs[par] = parfiles[par]

    # make sure certain values are integers
    for key in kwargs:
        if key in ["window", "bbminlength", "bbmaxlength"]:
            if np.isfinite(kwargs[key]):
                kwargs[key] = int(kwargs[key])

    # complex time series data
    data = dataset[()]

    # data time stamps (reconstructed from segments)
    segments = datasettimesegments[()]
    try:
        dt = kwargs.pop("dt")
    except KeyError:
        dt = kwargs.pop("dx")
    times = np.array([], dtype=float)
    for ts in segments:
        times = np.concatenate((times, np.arange(ts[0], ts[1] + dt / 2, dt)))
    kwargs["times"] = times

    filter_history = kwargs.pop("filter_history", None)

    # extract injection times if contained in the file
    injtimes = kwargs.pop("injtimes", None)

    if datasetstds is None:
        array = array_type(data, **kwargs)
    else:
        array = array_type(
            np.column_stack((data.real, data.imag, datasetstds[()])), **kwargs
        )

    # add filter history
    if filter_history is not None:
        array.filter_history = filter_history

    # re-add in injection times
    if injtimes is not None:
        array.injtimes = injtimes

        # add injection parameters as these will not be added when reading in
        # data that contains an injection
        if kwargs.get("injpar", None) is not None:
            array.injpar = kwargs["injpar"]
        else:
            array.injpar = array.par

    for par in ["par", "injpar"]:
        if par in parfiles:
            # remove temporary parameter file
            os.remove(parfiles[par])

    # set CWInPy version from read in file rather than current version
    array.cwinpy_version = kwargs["cwinpy_version"]

    # set heterodyne parameters
    array.include_ssb = kwargs.get("include_ssb", False)
    array.include_bsb = kwargs.get("include_bsb", False)
    array.include_glitch = kwargs.get("include_glitch", False)
    array.include_fitwaves = kwargs.get("include_fitwaves", False)

    # extract any Heterodyne arguments
    try:
        hetargs = io_hdf5.find_dataset(source, path=path + "HeterodyneArguments")
    except KeyError:
        hetargs = None

    if hetargs is not None:
        array.heterodyne_arguments = ast.literal_eval(hetargs[()][0].decode())

    # set any configuration file information
    if "cwinpy_heterodyne_dag_config" in kwargs:
        array.cwinpy_heterodyne_dag_config = kwargs["cwinpy_heterodyne_dag_config"]

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
    if series._input_stds:
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

    # remove metadata slots that can't/shouldn't be written as attributes
    slots = tuple()
    origslots = tuple(series._metadata_slots)
    badslots = [
        "par",
        "injpar",
        "laldetector",
        "running_median",
        "vars",
        "xindex",
        "heterodyne_arguments",
    ]

    if series._input_stds:
        # allow vars to be included
        badslots.remove("vars")

    for slot in series._metadata_slots:
        if slot not in badslots:
            slots += (slot,)
    series._metadata_slots = slots

    outseries = gwpy_write_hdf5_series(series, output, path=path, attrs=attrs, **kwargs)
    series._metadata_slots = origslots

    # add contiguous time segments (start and end times) as a different dataset
    # (path+"TimeSegments") in the HDF5 file. This allows non-contiguous time
    # series to be stored, but saves space compared to storing all time stamps
    dt = series.dt.value
    breaks = np.argwhere(np.diff(series.times.value) != dt)[:, 0].tolist()
    breaks = (
        [-1] + breaks + [len(series) - 1]
    )  # make sure first and last value are included
    segments = []
    for i in range(len(breaks) - 1):
        segments.append(
            (series.times.value[breaks[i] + 1], series.times.value[breaks[i + 1]])
        )
    write_metadata(segments, output, path=path + "TimeSegments", append=True)

    # add standard deviations to a different dataset (path+"Sigmas") in the HDF5 file
    if series._input_stds:
        write_metadata(series.stds, output, path=path + "Sigmas", append=True)

    # check heterodyne_arguments for segment list/frame cache list
    if series.heterodyne_arguments is not None:
        # output into a string
        write_metadata(
            [str(series.heterodyne_arguments)],
            output,
            path=path + "HeterodyneArguments",
            append=True,
        )

    return outseries


@io_hdf5.with_write_hdf5
def write_metadata(data, source, path=None, **kwargs):
    """
    Add metadata datasets to the HDF5 file. For example, non-consecutive time
    series time stamps.
    """

    return io_hdf5.create_dataset(source, path=path, data=data)


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
