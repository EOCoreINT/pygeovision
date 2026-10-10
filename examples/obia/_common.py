"""Shared setup for the OBIA tutorial scripts (kept tiny on purpose)."""
import os
import warnings

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")          # write PNG files; no window needed

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT, exist_ok=True)

# the rule set students are guided towards in session 3 (spectral rules first, then shape/size)
SPECTRAL_RULES = [("water", "mean_ndwi > 0.3"),
                  ("forest", "mean_ndvi > 0.55 and std_nir > 9"),
                  ("cropland", "mean_ndvi > 0.55"),
                  ("bare_field", "mean_ndvi <= 0.55")]
REFINED_RULES = [("water", "mean_ndwi > 0.3"),
                 ("road", "elongation > 6 and brightness < 110"),
                 ("building", "area_px < 400 and compactness > 0.5 and mean_ndvi <= 0.55"),
                 ("forest", "mean_ndvi > 0.55 and std_nir > 9 and area_px > 500"),
                 ("cropland", "mean_ndvi > 0.55"),
                 ("bare_field", "mean_ndvi <= 0.55")]
