#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 09:46:40 2025

@author: thomasgrandjean
"""

from .conftest import SAMPLE_DATASET
import geopandas as gpd

from netcdf2dataframe import netcdf2dataframe


def test_global():
    df = netcdf2dataframe(SAMPLE_DATASET, "tp")
    assert isinstance(df, gpd.GeoDataFrame)
