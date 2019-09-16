from numpy import (loadtxt, savetxt, column_stack)
from ..data import HeterodynedData

from gwpy.io import registry as io_registry
from gwpy.io.utils import identify_factory


# -- read ---------------------------------------------------------------------

def read_ascii_series(input_, array_type=HeterodynedData, **kwargs):
    """
    Read a `Series` from an ASCII file. This is a based on the
    :meth:`gwpy.types.io.ascii.read_ascii_series` function.

    Parameters
    ----------
    input : `str`, `file`
        file to read

    array_type : `type`
        desired return type
    """

    data = loadtxt(input_, **kwargs)

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

    try:
        comments = series.comments
    except AttributeError:
        comments = ""

    return savetxt(output, column_stack((xarr, yarrr, yarri)), header=comments, **kwargs)


# -- register -----------------------------------------------------------------


def register_ascii_series_io(array_type, format='txt', identify=True,
                             **defaults):
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
        io_registry.register_identifier(format, array_type,
                                        identify_factory(format))
