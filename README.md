# netcdf2dataframe
This repo contains a small utility used to transform netCDF files to geopandas.GeoDataFrames.

# Why netcdf2dataframe?

⚠️ This project was initiated due to the need of colleagues. It's considered a work in
progress !

# Installation

`pip install netcdf2dataframe`

# Basic usage

```
from netcdf2dataframe import netcdf2dataframe
path = "IMERG_total_precipitation_day_0.5x0.5_global_2021_v6.0.nc"
df = netcdf2dataframe(path, "pr")
```

Note that this module may end up crashing your machine, as DataFrames are not 
RAM-efficient objects (compared to the netCDF format). But sometimes, this is
what you may need...

Full docstring:

```
netcdf2dataframe(path: str, target: str = 'pr', dropna: bool = True) -> geopandas.geodataframe.GeoDataFrame
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
```

## Support

In case of bugs, please open an issue [on the repo](https://github.com/tgrandje/netcdf-to-geodataframe/issues).

## Contribution
Any help is welcome.

## Author
Thomas GRANDJEAN (DREAL Hauts-de-France, service Information, Développement Durable et Évaluation Environnementale, pôle Promotion de la Connaissance).

## Licence
GPL-3.0-or-later

## Project Status
Development.