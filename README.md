# Watching the Amazon Disappear from Space 🌳🛰️

I trained a deep learning model to find forest in satellite photos, then used it to spot where forest in the Brazilian Amazon was cut down between 2021 and 2026.

![Before and after](outputs/12_before_after.png)

*An 8 x 8 km piece of Rondônia, Brazil. On the left is July 2021, in the middle is July 2026, and on the right my model marks the forest that disappeared in red.*

## Why I made this

I kept reading news like "the Amazon lost X football fields of forest this year" and I always wondered: how does anyone actually know that? Who is counting the trees?

It turns out the answer is satellites. The European Space Agency has a satellite called Sentinel-2 that photographs the whole planet every few days, and the pictures are free for anyone. So I wanted to see if I could do the counting myself, with my own code, starting from raw satellite pictures and ending with a map that shows exactly where the forest went.

I picked Rondônia because it is one of the most deforested places on Earth. From above it looks like a fishbone: roads cut into the forest, and farms spread out from both sides.

## What I found

| | |
|---|---|
| Area I studied | about 55 x 55 km in Rondônia, Brazil |
| Forest cut down from 2021 to 2026 | **8,705 hectares**, about 7% of the forest that was there in 2021 |
| How big is that? | roughly **12,000 football fields** |
| Number of separate cleared patches | 1,762 (most are small, the biggest is 239 hectares) |
| Checked by hand on 300 random points | **87% overall accuracy** |
| Real cleared area estimated from those points | **about 9,800 hectares** (95% range 5,700 to 15,000), so the map's 8,705 holds up |
| How well my model finds forest | **91% IoU** against the ESA WorldCover forest map, on areas it never saw while learning |
| A simple "greenness" rule for comparison | 85% IoU |

IoU is a score for how well two maps overlap. 100% would mean my forest map matches WorldCover perfectly.

## How it works, in 5 steps

**1. Get the satellite pictures.**
I download Sentinel-2 pictures for July and August of 2021 and of 2026. I use the same months both times on purpose: in June the pasture is still green from the rain and looks a bit like forest, but by July it dries out, so forest and farmland are easy to tell apart. Clouds are a big problem in the Amazon, so I take 8 pictures for each year and, for every pixel, keep the middle value of the cloud free ones. That gives one clean, cloud free picture per year.

**2. Teach a model what forest looks like.**
For the answers, I use a free world map from the European Space Agency (WorldCover 2021) that already says which pixels are trees. I cut my 2021 picture into 441 small squares and trained a U-Net, a kind of neural network made for coloring in each pixel of an image. I trained it for free on a Kaggle GPU. To keep the test honest, the squares I tested on come from different parts of the map than the ones it learned from, so it can't just memorize its neighbors.

**3. Make both years look the same.**
The 2026 pictures came out a little brighter than 2021, not because the ground changed but because of haze and changes in how the satellite data is processed. So I found pixels that should never change, like old forest and town roofs, and used them to adjust the 2026 colors to match 2021.

**4. Map the forest in both years and compare.**
I ran the model over the whole area for each year, so every pixel gets a forest score from 0 to 100%. A pixel counts as cleared only if the model was confident it was forest in 2021 (60% or more) and confident it isn't forest in 2026 (40% or less). I also throw away tiny specks smaller than half a hectare, because those are usually just noise at the edges of the forest.

**5. Put it on a map.**
Every cleared patch becomes a red shape on an interactive map. You can click a patch to see how big it is, and switch between the 2021 and 2026 pictures to watch the forest vanish.

![Overview of the whole area](outputs/09_change_overview.png)

*The whole 55 x 55 km area. Red is forest that was cleared, green is forest that stayed. Blue is what the model thought was forest growing back, but my hand check showed that part is not reliable (see below).*

## Did I get it right?

I checked my results two ways: against a famous scientific map, and then with my own eyes.

### Check 1: a scientific map

First, I compared my results with Global Forest Change, a well known yearly map of forest loss made by scientists at the University of Maryland.

The good news: **94%** of what I marked as cleared, they also saw as cleared at some point. My model also agreed with them more than the simple greenness rule did.

The interesting part was where we disagreed. I looked at those spots one by one, and most of them turned out to be forest that **burned in 2024 but is still standing**. Their map counts any lost tree cover, while I was looking for forest turned into farmland, which is a different question. A few were real misses though, like a thin new logging road and a clearing that had only just started.

![Places my model missed](outputs/10_missed_check.png)

### Check 2: looking at 300 points myself

Comparing with another map only tells you how much two maps agree, and every map has its own mistakes. So I did what scientists do to measure a map properly: I picked 300 random points and looked at each one myself, in the 2021 and 2026 pictures, and wrote down what really happened there.

A few things I did to keep it honest:

* **I labelled blind.** My labelling page never showed what the model had said about a point, so I couldn't be nudged into agreeing with it.
* **I picked points from every part of the map.** Cleared land is only about 3% of the area, so 300 purely random points would have given me only about 9 cleared ones. Instead I picked 100 points from what the model called cleared, 80 from forest, 70 from non forest and 50 from regrowth, and then weighted each point by how much area it stands for. This is called stratified random sampling, and it's the standard method (Olofsson et al. 2014).

| | Result | 95% range |
|---|---|---|
| Overall accuracy | **87%** | 82% to 91% |
| Of what the model called cleared, how much really was | **64%** | 54% to 73% |
| Of all the real clearing, how much the model found | **57%** | 37% to 98% |
| Real cleared area | **9,769 ha** | 5,696 to 14,983 ha |

The best part: the model's 8,705 hectares sits right inside the range my own check found, so the main number holds up.

But the check also showed me things I would never have noticed otherwise:

* **The model sometimes thinks green crops, plantations and young bushy regrowth are forest.** When those get cut, it reports them as cleared forest. This was the biggest source of false alarms (32 of the 100 "cleared" points).
* **The regrowth map does not work.** Only 14 of the 50 regrowth points really had trees growing back. Most were just pasture that was greener in 2026. So I don't claim a regrowth number.
* **The range for "how much it found" is wide** because forest is huge, so each forest point stands for over 1,000 hectares, and a few misses swing the number a lot. More points along forest edges, where missed clearings hide, would narrow it.

You can see every point and my label in [`validation/`](validation/).

## Try the map yourself

Download [`outputs/deforestation_map.html`](outputs/deforestation_map.html) and open it in any browser. No setup needed.

## Run it yourself

```bash
pip install -r requirements.txt

python src/01_search.py              # find satellite pictures
python src/02_composite.py before    # build the clean 2021 picture
python src/02_composite.py after     # build the clean 2026 picture
python src/03_labels.py              # get the forest answers (WorldCover)
python src/04_tiles.py               # cut into training squares
python src/05_ndvi_baseline.py       # the simple greenness rule, to beat
# train the model: notebooks/forest-unet-rondonia.ipynb (I used a Kaggle GPU)
python src/07_normalize.py           # match 2026 brightness to 2021
python src/08_predict.py             # forest map for both years
python src/09_change.py              # find what was cleared
python src/10_validate_hansen.py     # compare with Global Forest Change
python src/11_map.py                 # build the interactive map
python src/13_sample_points.py       # pick 300 random points + labelling page
# label them in validation/labeler.html, save labels.csv into validation/
python src/14_accuracy.py            # accuracy and real area from your labels

# the SWIR experiment
python src/02_composite.py swir      # SWIR bands for both years
python src/04_tiles.py swir          # 6 band tiles (train in Kaggle, save to models/v2/)
python src/07_normalize.py           # also matches 2026 SWIR to 2021
python src/08_predict.py v2
python src/09_change.py v2
python src/14_accuracy.py v2
```

The trained model is already in `models/best_unet.pt`, so you can skip the training step.

## An experiment that didn't work (and what it taught me)

My hand check showed the model mixing up crops and young bushy regrowth with forest. I had a guess for why: it only sees normal colors and near infrared. Sentinel-2 also measures shortwave infrared (SWIR), which shows how much water is in leaves and soil, so forest canopy looks very different from dry crops and scrub.

So I downloaded the two SWIR bands, added two SWIR indices (NDMI for leaf moisture and NBR for burn scars), and retrained the exact same model on the exact same tiles. Then I scored the new map once on my same 300 points.

| | Without SWIR | With SWIR |
|---|---|---|
| IoU against WorldCover (test tiles) | 0.910 | 0.911 |
| Overall accuracy (my 300 points) | 86.6% | 86.1% |
| Of what it called cleared, how much really was | 64% | 58% |
| Of all real clearing, how much it found | 57% | 54% |

SWIR did fix part of what I aimed at: false alarms on crops and regrowth went from 32 to 25 points. But it also lost a few real clearings, and all the differences are well inside the uncertainty ranges. So honestly, no real improvement, and I kept the original model.

What I learned: the model learns whatever its answer key says is forest. WorldCover counts some plantations and bushy regrowth as tree cover, so the model copies that habit no matter how good its inputs are. The thing holding it back is the labels, not the satellite bands. I also stopped here on purpose: if I kept trying new versions against the same 300 points, I would slowly be tuning to my own test, and the numbers would stop meaning anything.

## What I want to try next

Train with better labels. MapBiomas, a Brazilian project, maps land cover every year and separates natural forest from plantations, pasture and young regrowth. Training on that, and checking it on a fresh set of hand labelled points, is the real test of my idea. I would also like to try it on other parts of the Amazon to see if it still works there.

## Built with

Python, PyTorch, Sentinel-2 imagery from Microsoft Planetary Computer, ESA WorldCover, Hansen Global Forest Change, GeoPandas, Rasterio and Folium.
