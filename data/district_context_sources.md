# District Context Data Sources

`district_context.csv` is a small hand-assembled district-level context table used
for exploratory analysis.

## Area and Population

- Districts 1, 3, 4, 5, 6, 7, and 8 use 2020 area and population values from
  Maison Office's HCMC district summary table.
- Districts 2 and 9 use former-district values because the housing dataset still
  uses old District 2 and District 9 labels. These districts were merged into
  Thu Duc City in 2021.

Useful references:

- HCMC administrative map and district table:
  https://maisonoffice.vn/en/news/hcmc-map/
- HCMC population table:
  https://maisonoffice.vn/en/news/ho-chi-minh-city-population/
- District 2 former district reference:
  https://en.wikipedia.org/wiki/District_2,_Ho_Chi_Minh_City
- District 9 former district reference:
  https://en.wikipedia.org/wiki/District_9,_Ho_Chi_Minh_City

## Coordinates

Latitude and longitude values are approximate district centroids used only to
calculate rough straight-line distance to District 1. They should be treated as
analysis features, not official administrative coordinates.

## Caveat

This file mixes current district data with former-district data for District 2
and District 9 so that it remains compatible with the original housing dataset.
For a publication-quality study, replace this with official GIS boundaries and
year-matched district-level demographic data from the HCMC Statistical Office or
General Statistics Office of Vietnam.
