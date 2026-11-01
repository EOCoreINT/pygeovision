# Revision notes for paper.md and paper.bib

Read this before sending the paper to Dr. Omid.

## What changed

* **New section "State of the field"** with a tool-by-tool comparison (Table 1: Earth Engine, openEO, Open Data Cube / Digital Earth Africa, sits, eo-learn, TorchGeo, TerraTorch, GeoAI/leafmap/samgeo, pyroSAR) and a table of design features (Table 2).
* **New section "Case studies compared with published work"** (Table 3): each of your eight case studies beside comparable published studies, with the difference that matters, and an explicit list of what cannot be claimed.
* **New section "Evaluation plan"** (Table 4): the quantitative comparisons that would turn the case studies into accuracy results (Sen1Floods11, MTBS, Landslide4Sense, GEO-Bench/PANGAEA, in-situ chlorophyll, another tool's streams, effort versus Earth Engine/GeoAI). **None has been run.** Dr. Omid will probably ask for at least one of them.
* **Two real figures** built from your actual outputs: the workflow diagram, and a panel of four report maps (Accra, Nepal, Tyrol HAND, London NDVI).
* **Limitations** and **testing** now say what is and is not verified.

## Claims I removed because the repository contradicts or does not support them

They are kept in a comment block at the end of paper.md, with the reason for each.

| Previous draft | Problem |
|---|---|
| GeoAgent, 86 % / 96 % routing accuracy | The agent module was removed from the package (see the roadmap). The numbers are not reproducible from the repository. |
| `pygeovision.insar`, the 170°/350° heading claim | No such module; InSAR is provided by pygeofetch. |
| 2.0 cm MAE against GPS (Kahramanmaraş) | No data or script in the repository. If it is yours, it belongs in the pygeofetch paper. |
| 847 ha Odaw flood in 18 h; 234.7 ha Atewa clearing | Not in the repository or the case studies. Add them back only as case studies with the same checks as the other eight. |
| Apache 2.0 | `LICENSE` and `pyproject.toml` say MIT. |
| 944+ tests, CI on three operating systems with a coverage gate | The repository has 2,285 collected tests. No `.github` folder was in the upload, so I cannot confirm CI. |
| 25 Jupyter notebooks | Not found in the repository. |
| 12 spectral indices | The README says 21. |
| Sample code (`fetch_and_preprocess`, `infer`, `export`) | Those methods do not exist in the client. It is replaced by the tested Accra pipeline call. |
| Old Figure 1 (Odaw flood map, InSAR map) | Shows results that are not documented in the repository. |

## Things only you can fix or confirm

1. **ORCID** is still the placeholder `0000-0000-0000-0000`.
2. **Version.** The paper says 2.2.0 (as in your draft and the crop-health report). The copy I worked on says 2.1.8. The test count (2,285) was taken from that copy; recount on the release tag with `pytest --collect-only -q | tail -1`.
3. **README says 77 architectures, the documentation says 79.** The paper uses 79. Make them agree.
4. **EOCoreINT "20 courses"** is your sentence, kept unchanged; I could not check it.
5. **Ghana EPA GIS Unit and NADMO** acknowledgement is left out until you confirm they agree to be named.
6. **pygeofetch entry** (`appiahkubi2025pygeofetch`) has no version or DOI. PyGeoVision requires 2.4.0 or later; archive a release on Zenodo and cite it.
7. **AI usage disclosure** now says Claude helped with this manuscript (checking against the repository, finding and verifying references, drafting the comparison sections). The previous text said it was grammar only, which is no longer true. Edit it if the facts differ.

## Bibliography

* **Fixed:** the old `jakubik2023prithvi` entry had the title "Prithvi-EO-2.0" but the 2023 arXiv number, which is the first Prithvi paper (*Foundation Models for Generalist Geospatial Artificial Intelligence*). It is corrected, and the real Prithvi-EO-2.0 paper (arXiv 2412.02732) is added as `szwarcman2024prithvi`.
* **Fixed:** the paper cited keys that were not in the .bib (`appiahkubi2026pygeofetch`, `chen2001`, `yaguemartinez2016`, `scheiber2000`, `berardino2002`, `jolivet2014`), and the .bib you uploaded was cut off after `terratorch2024` (the last line was `@so`). The InSAR references went with the InSAR claims. TerraTorch now cites its 2025 paper.
* **Duplicated or odd authors** in the geopandas entry were cleaned.
* **Every key used in the paper exists in the .bib, and every entry is cited** (checked by script; pandoc's citeproc raised no warning).
* **Checked online** (title, authors, venue and DOI read from the publisher, arXiv or archive page): Prithvi 1 and 2, TerraTorch, openEO, Open Data Cube (volume and pages not shown on the page), sits (first six authors), GeoAI, leafmap, geemap, samgeo, Landslide4Sense, Ghorbanzadeh 2019, Blaschke 2010 and 2014, Sen1Floods11, Twele 2016, Uddin 2019, Toffah and Pirotti 2026, Nobre 2011, Lindsay 2016, Eidenshink 2007, Dynamic World, WorldCover, Open Buildings, PANGAEA, GEO-Bench (first author), Moses 2012, ChangeFormer (arXiv number only).
* **From memory, not re-read online** (check the DOI): Mishra and Mishra 2012, McFeeters 1996, Xu 2006, Tucker 1979, Lu 2004, Barnes 2014, O'Callaghan and Mark 1984, Barrington-Leigh and Millard-Ball 2017 (title confirmed, other fields not), the Key and Benson authors, and the ChangeFormer IGARSS pages and DOI. Entries are marked in comments inside paper.bib.
* **pyroSAR** is cited as the 2019 conference poster, because no journal paper was found.

## The comparison with other tools

The "main job" column in Table 1 comes from each project's own documentation or paper, read in October 2026. A dash means I did not find it, not that it is impossible. Please check any row about a tool you know well before Dr. Omid does. One change from your draft matters: **GeoAI now searches and downloads Sentinel, Landsat and NAIP imagery**, so the old sentence that it "lacks data acquisition and preprocessing" would be rejected by a reviewer. The distinctive points are now the named pipelines with documented limits, scene selection that reports coverage, and the per-run record.

## Length and venue

The text is about 4,200 words including tables. The structure follows the Journal of Open Source Software (Summary, Statement of need, State of the field, Software design, Research impact, AI usage), whose guideline is much shorter (roughly 750 to 1,750 words). If the target is JOSS, move Tables 3 and 4 into the documentation and keep a paragraph each. For a longer journal paper the current length is fine. Ask Dr. Omid which venue he has in mind. I cited Blaschke's OBIA papers and the Salzburg landslide work because they are relevant to OBIA and to the evaluation plan; check with him that those are the ones he would cite.

## Honesty points a reviewer will test

* The case studies are **not accuracy assessments**, and the paper says so in bold.
* The F1 of 0.99 in the Accra report is agreement between radar and optical water in the dry window (how the radar threshold was calibrated), not accuracy.
* No test runs against live imagery, a GPU or downloaded pretrained weights.
* The Nepal rerun with `resolution_m=30` has not been tried.
