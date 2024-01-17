#!/usr/bin/env python3
"""Plot the data in the given directories."""
import argparse
import json
import os
import numpy as np
from scipy import stats  # type: ignore

# from scipy import norm  # type: ignore
from typing import List, Dict, Iterator, Callable, Any


class _Addrs:
    """Dictionary of addresses.

    The key is the address, and the value is the pretty name to use in the plot.
    """

    def __init__(self, addresses: Dict[str, str]):
        """Initialize the data point."""
        self._addresses = addresses

    def __getitem__(self, key: str) -> str:
        """Get the data for the given key."""
        return self._addresses[key]

    def addrs(self) -> List[str]:
        """Return the tests."""
        return list(self._addresses.keys())


class _Correlation:
    """Dictionary of correlation coefficients.

    The key is the test, and the value is the correlation coefficient.
    """

    def __init__(self, correlation: Dict[str, float]):
        """Initialize the data point."""
        self.correlation = correlation

    def __getitem__(self, key: str) -> float:
        """Get the data for the given key."""
        return self.correlation[key]

    def tests(self) -> List[str]:
        """Return the tests."""
        return list(self.correlation.keys())


class _JSONDataPoints:
    """List of JSON objects."""

    def __init__(self, objects: List[Dict[str, str]]):
        """Initialize the data point."""
        self.objects = objects

    def __iter__(self) -> Iterator[Dict[str, str]]:
        """Return an iterator."""
        return iter(self.objects)

    def __getitem__(self, key: int) -> Dict[str, str]:
        """Get the data for the given key."""
        return self.objects[key]


class _RawData:
    """Dictionary of JSONDataPoints.

    The key is the name of the test, and the value is the JSONDataPoints object.
    """

    def __init__(self, data: Dict[str, _JSONDataPoints]):
        """Initialize the data point."""
        self.data = data

    def __getitem__(self, key: str) -> _JSONDataPoints:
        """Get the data for the given key."""
        return self.data[key]

    def tests(self) -> List[str]:
        """Return the tests."""
        return list(self.data.keys())


class _DataPoint:
    """Data point for plotting.

    Just the x and y values.
    """

    def __init__(self, x: float, y: float):
        """Initialize the data point."""
        self.x = x
        self.y = y


class _ParsedDataAddrs:
    """Dictionary of DataPoints.

    Holds the DataPoints for each address.
    """

    def __init__(self, data: Dict[str, List[_DataPoint]]):
        """Initialize the data point."""
        self.data = data

    def __getitem__(self, addr: str) -> List[_DataPoint]:
        """Get the data for the given key."""
        return self.data[addr]

    def addrs(self) -> List[str]:
        """Return the tests."""
        return list(self.data.keys())


class _ParsedData:
    """Dictionary of DataPoints.

    Holds the _ParsedDataAddrs for each test.
    """

    def __init__(self, data: Dict[str, _ParsedDataAddrs]):
        """Initialize the data point."""
        self.data: Dict[str, _ParsedDataAddrs] = data

    def __getitem__(self, test: str) -> _ParsedDataAddrs:
        """Get the data for the given key."""
        return self.data[test]

    def __iter__(self) -> Iterator[str]:
        """Return an iterator."""
        return iter(self.data.keys())

    def tests(self) -> List[str]:
        """Return the tests."""
        return list(self.data.keys())

    def y_values(self, test: str, addr: str) -> List[float]:
        """Return the y values for the given test and address."""
        return [datapoint.y for datapoint in self.data[test][addr]]

    def set_y_values(self, test: str, addr: str, values: List[float]) -> None:
        """Set the y values for the given test and address."""
        for i, obj in enumerate(self.data[test][addr]):
            obj.y = values[i]


class _Correlation_collection:
    """Dictionary of datasets.

    The key is the name of the dataset, and the value is the _Correlation
    The key is typically the directory name.
    """

    def __init__(self, data: Dict[str, _Correlation]):
        """Initialize the data point."""
        self.data = data

    def __getitem__(self, key: str) -> _Correlation:
        """Get the data for the given key."""
        return self.data[key]

    def datasets(self) -> List[str]:
        """Return the tests."""
        return list(self.data.keys())


def _parse_files(files: Dict[str, str], tests: Dict[str, str]) -> _RawData:
    """Parse the data in the given directories.

    Args:
        files:      List of files to parse. The files should be one for each
                    test as specified in tests.
        tests:      Dictionary of tests to parse for each directory. The key is
                    name of the test as used in the directory name, and the
                    value is the pretty name to use in the plot.
        addresses:  Dictionary of addresses to parse for each directory. The key
                    is the address as used in the directory name, and the value
                    is the pretty name to use in the plot.

    Returns:
        A list of JSON objects.
    """
    data: Dict[str, _JSONDataPoints] = {}
    for test in files.keys():
        file_data: List[Dict[str, str]] = []
        with open(files[test], "r") as fp:
            for line in fp:
                try:
                    file_data.append(json.loads(line))
                except json.decoder.JSONDecodeError:
                    continue
        data[test] = _JSONDataPoints(file_data)

    return _RawData(data)


def _get_files(directory: str, tests: Dict[str, str]) -> Dict[str, str]:
    """Get the files to parse."""
    files: Dict[str, str] = {}
    subdir: str = directory.split("/")[-1]
    for test in tests.keys():
        if not os.path.exists(f"{directory}/{subdir}_{test}/"):
            continue
        for file in os.listdir(f"{directory}/{subdir}_{test}/"):
            if file.endswith(".log"):
                files[test] = f"{directory}/{subdir}_{test}/{file}"

    return files


def _parse_data(data: _RawData, addrs: _Addrs, field: str) -> _ParsedData:
    """Parse the data."""
    parsed_data: _ParsedData = _ParsedData({})

    for test in data.tests():
        owd_offset1: float = 0
        owd_offset2: float = 0

        data_points: _ParsedDataAddrs = _ParsedDataAddrs(
            {addr: [] for addr in addrs.addrs()}
        )
        for obj in data[test]:
            if obj[ip_side] in addrs.addrs() and obj[field] is not None:
                # NOTE: this is a hack to make the OWD calculate the actual
                #       values from the diffs generated by nethint
                # if field == json_field:
                if False:
                    if obj[ip_side] == addrs.addrs()[0]:
                        owd_offset1 += float(obj[field])
                        data_points[obj[ip_side]].append(
                            _DataPoint(
                                float(obj["rel_time"]),
                                owd_offset1,
                            )
                        )
                    else:
                        owd_offset2 += float(obj[field])
                        data_points[obj[ip_side]].append(
                            _DataPoint(
                                float(obj["rel_time"]),
                                owd_offset2,
                            )
                        )

                else:
                    data_points[obj[ip_side]].append(
                        _DataPoint(float(obj["rel_time"]), float(obj[field]))
                    )

        parsed_data.data[test] = data_points

    return parsed_data


def _plot_data(
    data: _ParsedData,
    tests: Dict[str, str],
    addrs: _Addrs,
    field: str,
    plot_type: str | None = None,
    title: str | None = None,
    y_label: str | None = "OWD (ms)",
    x_label: str | None = "Time (s)",
    correlation: _Correlation | None = None,
    filename: str | None = None,
) -> None:
    """Plot the data."""
    import matplotlib.pyplot as plt
    import scienceplots  # type: ignore
    import random
    import math

    plt.style.use(["science", "grid", "ieee"])
    # plt.rcParams.update({"legend.fontsize": "small", "axes.labelsize": "large"})
    plt.rcParams.update({"axes.labelsize": "x-large", "legend.framealpha": "0.85"})

    filenames: List[str] | None = None
    if filename is not None:
        filenames = []
        for test in data.tests():
            filenames.append(
                f"{filename.split('.')[0]}-{test}.{filename.split('.')[-1]}")

    for i, test in enumerate(sorted(data.tests())):
        # ax = plt.subplot(len(data.tests()), 1, i + 1)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.grid()

        def _plot_data_points(ax: Any, test: str):
            for addr in addrs.addrs():
                x_values = []
                y_values = []
                for data_point in data[test][addr]:
                    x_values.append(data_point.x)
                    y_values.append(data_point.y)

                match plot_type:
                    case "histogram":
                        ax.hist(y_values, label=f"{addrs[addr]}")
                    case "cdf":
                        ax.plot(
                            x_values,
                            y_values,
                            "o",
                            markersize=0.7,
                            label=f"{addrs[addr]}",
                        )
                    case None:
                        ax.plot(
                            x_values,
                            y_values,
                            "o",
                            markersize=0.7,
                            label=f"{addrs[addr]}",
                        )

        def _plot_boxplot(ax: Any, test: str) -> None:
            for i, addr in enumerate(addrs.addrs()):
                y_values = [data_point.y for data_point in data[test][addr]]
                ax.boxplot(
                    y_values,
                    labels=[addrs[addr]],
                    positions=[i],
                )

        match plot_type:
            case "boxplot":
                _plot_boxplot(ax, test)
            case _:
                _plot_data_points(ax, test)
                ax.legend()

        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)

        if filenames is not None:
            plt.savefig(filenames[i], dpi=700)
            plt.close()
        else:
            plt.show()


def _plot_correlation(
    data: _Correlation_collection,
    tests: Dict[str, str],
    title: str | None = None,
    filename: str | None = None,
    legend_loc: str | None = None,
) -> None:
    """Plot a CDF graph of the correlation coefficients."""
    import matplotlib.pyplot as plt
    import scienceplots  # type: ignore
    import math

    plt.style.use(["science", "grid", "ieee"])
    # plt.rcParams.update({"legend.fontsize": "small", "axes.labelsize": "large"})
    plt.rcParams.update({"axes.labelsize": "x-large", "legend.framealpha": "0.85"})

    fig, ax = plt.subplots(nrows=1, ncols=1)
    fig.subplots_adjust(hspace=1)

    for testcase in sorted(
        tests.keys(), key=lambda k: tests[k]
    ):  # Testcase is common, nocommon, wifi
        ax = plt.subplot(1, 1, 1)

        y_values: List[float] = []
        for directory in data.datasets():
            if testcase not in data[directory].correlation:
                continue
            if not math.isnan(data[directory].correlation[testcase]):
                y_values.append(data[directory].correlation[testcase])

        # Get CDF
        x = np.sort(y_values)
        y = np.arange(1, len(x) + 1) / len(x)

        ax.plot(x, y, label=tests[testcase])  # "o", markersize=0.7)

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel("Correlation coefficient")
    ax.set_ylabel("CDF")

    if legend_loc is not None:
        ax.legend(loc=legend_loc)
    else:
        ax.legend()
    # ax.grid()

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def _calculate_correlation(
    data: _ParsedData,
    tests: Dict[str, str],
    addrs: _Addrs,
    field: str,
    directory: str,
) -> _Correlation:
    """Calculate the correlation coefficient between the devices."""
    correlation: Dict[str, float] = {}
    for test in data.tests():
        x1_values: List[float] = []
        y1_values: List[float] = []
        x2_values: List[float] = []
        y2_values: List[float] = []
        for addr in addrs.addrs():
            for data_point in data[test][addr]:
                if addr == addrs.addrs()[0]:
                    x1_values.append(data_point.x)
                    y1_values.append(data_point.y)
                else:
                    x2_values.append(data_point.x)
                    y2_values.append(data_point.y)

        if len(x1_values) == 0 or len(x2_values) == 0:
            print(f"{directory}" f"-{test}" f": no data (correlation)")
            continue

        # interpolate the data so that they have the same x values
        x1 = np.array(x1_values)
        y1 = np.array(y1_values)
        x2 = np.array(x2_values)
        y2 = np.array(y2_values)
        if len(x1) > len(x2):
            y2 = np.interp(x1, x2, y2)
        else:
            y1 = np.interp(x2, x1, y1)

        # calculate the correlation coefficient
        corr, _ = stats.pearsonr(y1, y2)
        correlation[test] = corr
        print(f"correlation: {directory}-{test}: {corr}")

    return _Correlation(correlation)


def _get_cdf(
    data: _ParsedData, tests: Dict[str, str], addrs: _Addrs, field: str
) -> _ParsedData:
    """Plot the CDF of the data."""
    cdf_data: _ParsedData = _ParsedData({})
    for test in data.tests():
        for addr in addrs.addrs():
            y_values = [datapoint.y for datapoint in data[test][addr]]

            x = np.sort(y_values)
            y = np.arange(1, len(x) + 1) / len(x)

            # put the new data into `data`
            for i in range(len(x)):
                if test not in cdf_data.data:
                    cdf_data.data[test] = _ParsedDataAddrs({})
                if addr not in cdf_data.data[test].data:
                    cdf_data.data[test].data[addr] = []
                cdf_data.data[test].data[addr].append(_DataPoint(x[i], y[i]))

    return cdf_data


def make_graph(
    directories: List[str],
    tests: Dict[str, str],
    addresses: Dict[str, str],
    ip_side: str = "ip_dst",
    field: str = "rtt",
    noplot: bool = False,
    verbose: bool = False,
    correlation: bool = False,
    plot_type: str | None = None,
    title: str | None = None,
    filename: str | None = None,
    filterer: Callable[[_DataPoint], bool] | None = None,
    maper: Callable[[_DataPoint], _DataPoint] | None = None,
    performer: Callable[[_ParsedData], _ParsedData] | None = None,
    finalcdf: bool = False,
    legend_loc: str | None = None,
):
    """Plot the data in the given directories.

    Args:
        directories: List of directories to plot.
        tests:       Dictionary of tests to plot for each directory. The key is
                     name of the test as used in the directory name, and the
                     value is the pretty name to use in the plot.
        addresses:   Dictionary of addresses to plot for each directory. The key
                     is the address as used in the directory name, and the value
                     is the pretty name to use in the plot.
        field:       The field to plot.
        noplot:      Don't plot the data.
        verbose:     Print the number of datapoints for each address.
        correlation: Calculate the correlation coefficient.
        cdf:         Plot the CDF of the data.
        title:       Title of the plot.
        filename:    Filename of the plot.
        filters:     Dictionary of filters to apply to the data. The key is the
                     address, and the value is the filter function.
        maper:       Dictionary of map functions to apply to the data. The key
                     is the address, and the value is the map function.
        performer:   Function to perform on the data. The function should take
                     _RawData as input and return _RawData as output.
        finalcdf:    Plot the final CDF graph.
    """
    addrs = _Addrs(addresses)
    ip_side = ip_side

    if finalcdf:
        _filename = None
        noplot = True
    else:
        _filename = filename

    datasets: _Correlation_collection = _Correlation_collection({})

    for directory in directories:
        files: Dict[str, str] = _get_files(directory, tests)
        data: _RawData = _parse_files(files, tests)
        parsed_data: _ParsedData = _parse_data(data, addrs, field)
        correlations: _Correlation | None = None

        if filterer:
            # filter the data
            for test in data.tests():
                for addr in addrs.addrs():
                    parsed_data.data[test].data[addr] = list(
                        filter(filterer, parsed_data.data[test].data[addr])
                    )

        if maper:
            # map the data
            for test in data.tests():
                for addr in addrs.addrs():
                    parsed_data.data[test].data[addr] = list(
                        map(maper, parsed_data.data[test].data[addr])
                    )

        if performer:
            # perform the operation on the data
            parsed_data = performer(parsed_data)

        if plot_type == "cdf":
            parsed_data = _get_cdf(parsed_data, tests, addrs, field)

        # calculate the correlation coefficient
        if correlation or finalcdf:
            correlations = _calculate_correlation(
                parsed_data, tests, addrs, field, directory
            )

        # print number of datapoints for each address
        if verbose:
            for test in data.tests():
                for addr in addrs.addrs():
                    print(
                        "num_datapoints: "
                        f"{directory}"
                        f"-{test}"
                        f"-{addr}"
                        f" {len(parsed_data[test][addr])}"
                    )

        # plot the data
        match plot_type:
            case "cdf":
                y_label: str | None = "CDF"
                x_label: str | None = f"{field.upper()} (ms)"
            case "boxplot":
                y_label = f"{field.upper()} (ms)"
                x_label = None
            case "histogram":
                y_label = "Frequency"
                x_label = f"{field.upper()} (ms)"
            case _:
                y_label = f"{field.upper()} (ms)"
                x_label = "Time (s)"

        if not noplot:
            _plot_data(
                parsed_data,
                tests,
                addrs,
                field,
                plot_type=plot_type,
                title=title,
                correlation=correlations,
                filename=_filename,
                y_label=y_label,
                x_label=x_label,
            )

        if finalcdf and correlations is not None:
            datasets.data[directory] = correlations

    if finalcdf:
        _plot_correlation(datasets, tests, title, filename, legend_loc)


def _parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Python plotter")
    parser.add_argument("directories", nargs="+", help="directories to plot")
    parser.add_argument("-n", "--noplot", action="store_true", help="don't plot")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    parser.add_argument("-c", "--correlation", action="store_true", help="correlation")
    parser.add_argument(
        "-f",
        "--field",
        action="store",
        type=str,
        help="field to plot",
        default="rtt",
    )
    parser.add_argument(
        "-t",
        "--title",
        action="store",
        type=str,
        help="title of the plot",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--output",
        action="store",
        type=str,
        help="output file",
        default=None,
    )
    parser.add_argument(
        "-e",
        "--extension",
        action="store",
        type=str,
        help="output file extension",
    )
    parser.add_argument(
        "-l",
        "--legend",
        action="store",
        type=str,
        help="legend location",
    )
    parser.add_argument(
        "-C", "--cdf", action="store_true", help="plot CDF graph"
    )

    command_group = parser.add_mutually_exclusive_group()
    command_group.add_argument(
        "-B",
        "--boxplot",
        action="store_true",
        help="plot boxplot instead of line graph",
    )
    command_group.add_argument(
        "-H",
        "--histogram",
        action="store_true",
        help="plot histogram instead of line graph",
    )
    command_group.add_argument(
        "-F",
        "--finalcdf",
        action="store_true",
        help="plot final CDF graph",
    )

    args = parser.parse_args()

    for i, file in enumerate(args.directories):
        if file.endswith("/"):
            args.directories[i] = args.directories[i][:-1]

    return args


if __name__ == "__main__":
    args = _parse_args()

    if args.output is not None:
        filename = args.output
    elif args.extension is not None and args.title is not None:
        filename = f"{args.title}.{args.extension}"
    elif args.extension is not None:
        filename = f"{args.directories[0].split('/')[-1]}.{args.extension}"
    else:
        filename = None

    plot_type: str | None = None
    if args.boxplot:
        plot_type = "boxplot"
    elif args.cdf:
        plot_type = "cdf"
    elif args.histogram:
        plot_type = "histogram"

    legend_loc: str | None = None
    if args.legend is not None:
        legend_loc = args.legend

    tests = {
        "common": "Common bottleneck\nat router",
        "nocommon": "No common bottleneck",
        "wifi": "Common bottleneck\nat WiFi AP",
    }

    # testcase = "router"
    testcase = "clients"

    match testcase:
        case "router":
            addresses = {"10.10.12.1": "Source 1", "172.16.11.3": "Source 2"}
            ip_side = "ip_dst"
            # ip_side = "ip_src"
        case "clients":
            # do something for case2
            addresses = {"172.16.13.3": "Source 1", "172.16.13.4": "Source 2"}
            # addresses = { "172.16.13.4": "Source 2"}
            # addresses = {"172.16.13.3": "Source 1"}
            ip_side = "ip_dst"

    # addresses = {"172.16.12.4": "pc04", "172.16.12.5": "pc05"}
    # addresses = {"172.16.13.3": "Source 1", "172.16.13.4": "Source 2"}
    # addresses = {"10.10.12.1": "Source 1", "172.16.11.3": "Source 2"}
    # addresses = {"10.10.12.1": "Source 1"}

    # whether to use 'ip_dst' or 'ip_src' as the filter
    # ip_side = "ip_dst"
    # ip_side = "ip_src"

    # filterer = None
    def filterer(x: _DataPoint) -> bool:
        """Filter out data points."""
        # return x.y > 1 or x.y < -1
        return x.y != 0
        # return 0 != x.y < 3
        return True

    # maper = None
    def maper(x: _DataPoint) -> _DataPoint:
        """Map the data points."""
        # return _DataPoint(x.x, abs(x.y))
        return x

    def performer(data: _ParsedData) -> _ParsedData:
        """Perform an operation on the data."""
        # Shift the data so that the minimum value is 0
        for test in data.tests():
            for addr in addresses:
                y_values: List[float] = data.y_values(test, addr)
                min_y = min(y_values)
                y_values = [y - min_y for y in y_values]
                data.set_y_values(test, addr, y_values)

        return data

    make_graph(
        args.directories,
        tests,
        addresses,
        ip_side,
        noplot=args.noplot,
        verbose=args.verbose,
        field=args.field,
        correlation=args.correlation,
        plot_type=plot_type,
        title=args.title,
        filename=filename,
        filterer=filterer,
        maper=maper,
        performer=performer,
        finalcdf=args.finalcdf,
        legend_loc=legend_loc,
    )
