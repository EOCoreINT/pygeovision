# Chapter 25: Urban

## 25.1 Building Footprint Extraction

```python
result = client.pipeline("building_footprints",
                          bbox=(-0.30, 5.50, -0.05, 5.70),  # Accra
                          date="2026-06",
                          output_dir="./urban/accra/")

buildings = gpd.read_file(result.output_path)
print(f"Buildings: {len(buildings):,}")
print(f"Total footprint: {buildings.area.sum()/1e4:.0f} ha")
print(f"Mean area: {buildings.area.mean():.0f} m2")
```

## 25.2 Urban Growth Analysis

```python
result = client.pipeline("urban_growth",
                          bbox=BBOX, date="2026-06", output_dir="./urban/")
growth = result.stats

print(f"New urban area (2016-2026): {growth.get('new_urban_km2', 0):.1f} km2")
print(f"Annual growth rate: {growth.get('growth_rate_pct', 0):.2f}%/year")
```

## 25.3 Infrastructure Monitoring

```python
# Track construction progress over time
pre_buildings  = gpd.read_file("./urban/buildings_2024.geojson")
post_buildings = gpd.read_file("./urban/buildings_2026.geojson")

new_buildings  = post_buildings[~post_buildings.geometry.within(
    pre_buildings.unary_union.buffer(5)
)]
print(f"New buildings (2024-2026): {len(new_buildings):,}")
```

## 25.4 Road Network Extraction

```python
result = client.pipeline("road_network",
                          bbox=BBOX, date="2026-06", output_dir="./roads/")
roads = gpd.read_file(result.output_path)
print(f"Total road length: {roads.length.sum()/1000:.0f} km")
```

## Exercises

1. Extract building footprints for your city centre (1x1 km).
2. Compare building density (buildings/km2) between urban and suburban areas.
3. Map 10-year urban growth using change detection.

## 25.7 Summary

Urban monitoring with PyGeoVision covers building extraction, growth analysis,
road networks, and night-time lights. Fine resolution data (< 3m) significantly
improves individual building accuracy.

## Exercises

1. Extract building footprints for a city centre (1x1 km).
2. Compare building density between formal and informal areas.
3. Map 10-year urban growth using change detection.
4. Compute urban heat island for 3 districts.
