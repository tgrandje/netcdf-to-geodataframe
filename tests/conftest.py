#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 10:15:56 2025

@author: thomasgrandjean
"""

import cdsapi
import os
import zipfile

DIRNAME = os.path.dirname(__file__)

SAMPLE_DATASET = os.path.join(DIRNAME, "sample.nc")


def pytest_sessionstart(session):
    """
    download sample dataset from CDS
    https://cds.climate.copernicus.eu/datasets/sis-energy-pecd?tab=download
    """

    dataset = "sis-energy-pecd"
    request = {
        "pecd_version": "pecd4_1",
        "temporal_period": ["historical", "future_projections"],
        "origin": ["era5_reanalysis", "cmcc_cm2_sr5"],
        "emissions": ["ssp2_4_5"],
        "variable": ["total_precipitation"],
        "spatial_resolution": ["0_25_degree", "nuts_2"],
        "year": ["2020"],
        "month": ["12"],
    }

    client = cdsapi.Client(
        url=os.environ.get("CDS_URL"), key=os.environ.get("CDS_KEY")
    )
    path = os.path.join(DIRNAME, "sample.zip")
    client.retrieve(dataset, request, target=path)
    with zipfile.ZipFile(path) as z:
        files = [x for x in z.namelist() if os.path.splitext(x)[-1] == ".nc"]
        file = files[0]
        with z.open(file, "r") as netsample:
            with open(SAMPLE_DATASET, "wb") as f:
                f.write(netsample.read())
    os.remove(path)


def pytest_sessionfinish(session, exitstatus):
    os.remove(SAMPLE_DATASET)
