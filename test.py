import pygeovision as pgv

client = pgv.PyGeoVision()
scene = client.classification.land_cover(
    provider="copernicus",
    bbox=[-0.30, 5.50, -0.10, 5.70],   # Odaw River Basin
    date="2024-06-15",
    sensor="S1_GRD",
)
flood = client.infer(scene, model="prithvi_eo_2_0", task="flood_detection")
client.export(flood, "odaw_flood_2024-06-15.geojson")