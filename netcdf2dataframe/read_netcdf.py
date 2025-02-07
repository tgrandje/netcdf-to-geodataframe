#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:59:31 2025

@author: thomasgrandjean
"""

from cftime import num2pydate
from functools import lru_cache
import logging
import sys
from typing import Tuple


import geopandas as gpd
from netCDF4 import Dataset, Variable
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=None)
def warning_map_griding(grid_mapping: str):
    logger.error(
        "grid_mapping if ignored for now: coordinates are expected to be using"
        " EPSG:4326 system. Found grid_mapping=%s",
        grid_mapping,
    )


class Dataset2DataFrame(Dataset):
    """
    Customized netCDF4.Dataset with some methods added to convert the netCDF
    file to GeoDataFrames
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._parse_times()
        self._parse_coordinates()

    def get(self, attribute):
        "try to get a given attribute, returns None if not available"
        try:
            return getattr(self, attribute)
        except AttributeError:
            return None

    @staticmethod
    def set_series_index(s, index_name):
        "converts a series to"
        s.index.name = index_name
        s = s.reset_index(drop=False)
        return s

    def _parse_times(self) -> pd.DataFrame:
        """
        Evaluate the timeseries set in the netCDF and store it in self.timeseries

        Times are indexed by their order of value in the netCDF structure

        Returns
        -------
        times : pd.DataFrame

        Ex.:
            time   time_val
         0     0 2021-01-01
         1     1 2021-01-02
         2     2 2021-01-03
         3     3 2021-01-04
         4     4 2021-01-05


        """
        times = self.variables["time"]
        times = num2pydate(
            times[:], units=times.units, calendar=times.calendar
        )
        times = self.set_series_index(
            pd.Series(times, name="time_val"), "time"
        )
        return times

    def _parse_coordinates(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Evaluate the coordinates set in the netCDF and store it in self.lats
        and self.longs

        Note that grid_mapping are (for now) ignored: this is working under the
        assumption that coordinates use the 4326 EPSG.

        Lats/longs are indexed by their order of value in the netCDF structure

        Returns
        -------
        lats, longs : Tuple[pd.DataFrame, pd.DataFrame]

        First element is a dataframe of latitudes, the second contains
        longitudes

        Ex. :
            lat    lat_val
         0    0 -89.749992
         1    1 -89.249992
         2    2 -88.749992
         3    3 -88.249992
         4    4 -87.749992

         Ex. :
            long    long_val
         0     0 -179.749985
         1     1 -179.249985
         2     2 -178.749985
         3     3 -178.249985
         4     4 -177.749985

        """

        lats = self.variables["lat"]
        if lats.units not in {
            "degrees_north",
            "degree_north",
            "degree_N",
            "degrees_N",
            "degreeN",
            "degreesN",
        }:
            raise ValueError(f"unexpected unit for latitudes: {lats.units}")
        lats = self.set_series_index(pd.Series(lats[:], name="lat_val"), "lat")

        longs = self.variables["lon"]
        if longs.units not in {
            "degrees_east",
            "degree_east",
            "degree_E",
            "degrees_E",
            "degreeE",
            "degreesE",
        }:
            raise ValueError(f"unexpected unit for longitudes: {longs.units}")
        longs = self.set_series_index(
            pd.Series(longs[:], name="long_val"), "long"
        )

        for arg in [lats, longs]:
            try:
                warning_map_griding(longs.grid_mapping)
            except AttributeError:
                continue

        return lats, longs

    def to_dataframe(
        self, target: str = "pr", dropna: bool = True
    ) -> gpd.GeoDataFrame:
        """
        Extract the target from the netCDF and convert it to a GeoDataFrame
        format. The obtained GeoDataFrame should have a geometry column, a
        timeseries column and a column representing the desired target.

        Parameters
        ----------
        target : str, optional
            The desired variable. The default is "pr".
        dropna : TYPE, optional
            If True, the GeoDataFrame will not contain missing values, hence
            being more frugal (less RAM consumption). The default is True.

        Returns
        -------
        df : gpd.GeoDataFrame
            GeoDataFrame des données

            Ex.:
                        value   time_val                      geometry
              0  3.000000e+33 2021-01-01  POINT (-179.74998 -89.74999)
              1  3.000000e+33 2021-01-01  POINT (-179.24998 -89.74999)
              2  3.000000e+33 2021-01-01  POINT (-178.74998 -89.74999)
              3  3.000000e+33 2021-01-01  POINT (-178.24998 -89.74999)
              4  3.000000e+33 2021-01-01  POINT (-177.74998 -89.74999)

        """

        times = self._parse_times()
        lats, longs = self._parse_coordinates()
        variable = self.variables[target]

        df = self._unstack_array(variable, dropna)

        size = sys.getsizeof(df) / 1024**3
        logger.warning("raw df is %sGb in RAM", round(size, 2))

        df = df.merge(times, on="time", how="left").drop("time", axis=1)
        df = df.merge(lats, on="lat", how="left").drop("lat", axis=1)
        df = df.merge(longs, on="long", how="left").drop("long", axis=1)

        x, y = "long_val", "lat_val"
        geometries = df[[x, y]].drop_duplicates()
        geometries["geometry"] = gpd.points_from_xy(
            x=geometries[x], y=geometries[y], crs=4326
        )
        df = df.merge(geometries, on=[x, y], how="inner").drop([x, y], axis=1)
        df = gpd.GeoDataFrame(df, geometry="geometry", crs=4326)

        size = sys.getsizeof(df) / 1024**3
        logger.warning("df is %sGb in RAM", round(size, 2))

        df = df.rename({"value": target}, axis=1)

        return df

    @staticmethod
    def _unstack_array(
        variable: Variable, dropna: bool = True
    ) -> pd.DataFrame:
        """
        Inner function reshaping the 3D data to an unstacked DataFrame

        Parameters
        ----------
        variable : netCDF4.Variable
            Variable you want to extract (and reshape)
        dropna : bool, optional
            If True, the DataFrame will not contain missing values, hence
            being more frugal (less RAM consumption). The default is True.

        Returns
        -------
        df : pd.DataFrame

        Ex.:
                time  lat  long     value
         16560     0   23     0  0.000000
         16565     0   23     5  0.382080
         16566     0   23     6  1.625944
         16567     0   23     7  1.772459
         16568     0   23     8  1.412688

        """

        data = variable[:]

        # data is 3D:
        # dim1 is time
        # dim2 is latitude
        # dim3 is longitude

        indices = np.indices(data.shape)
        reshaped_indices = indices.reshape(3, -1).T
        reshaped_values = data.reshape(-1, 1)
        reshaped_mask = data.mask.reshape(-1, 1)

        df = pd.DataFrame(
            np.hstack((reshaped_indices, reshaped_values, reshaped_mask)),
            columns=["time", "lat", "long", "value", "mask"],
        )

        if dropna:
            ix = df[df["mask"] == 1].index
            df = df.drop(ix)
        df = df.drop("mask", axis=1)

        for f in ["time", "lat", "long"]:
            df[f] = df[f].astype(np.int32)

        return df


def netcdf2dataframe(
    path: str, target: str = "pr", dropna: bool = True
) -> gpd.GeoDataFrame:
    """
    Picks a netCDF file and extracts one variable ("target") on the form of
    a GeoDataFrame.

    Parameters
    ----------
    path : str
        Path to netCDF file
    target : str, optional
        Target variable. The default is "pr".
        Note that available variables are displayed in the log (info level).
    dropna : bool, optional
        If True, the GeoDataFrame will not contain missing values, hence
        being more frugal (less RAM consumption). The default is True.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame of the extracted values

    Example
    -------
    >>> df = netcdf2dataframe(
        "IMERG_total_precipitation_day_0.5x0.5_global_2021_v6.0.nc,
        "pr",
        )
    >>> df.head()

             pr   time_val                      geometry
    0  0.000000 2021-01-01  POINT (-179.74998 -78.24999)
    1  0.382080 2021-01-01  POINT (-177.24998 -78.24999)
    2  1.625944 2021-01-01  POINT (-176.74998 -78.24999)
    3  1.772459 2021-01-01  POINT (-176.24998 -78.24999)
    4  1.412688 2021-01-01  POINT (-175.74998 -78.24999)


    """

    net = Dataset2DataFrame(path, "r", format="NETCDF4")

    for attribute in "description", "history", "source":
        logger.info("%s: %s", attribute, net.get(attribute))
    logger.info("variables: %s", net.variables.keys())
    return net.to_dataframe(target, dropna)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    path = "IMERG_total_precipitation_day_0.5x0.5_global_2021_v6.0.nc"
    df = netcdf2dataframe(path, "pr")
