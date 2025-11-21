# Finetuned YOLO v11L on custom annotated dataset with 71 brand logos, for KI Sports hackathon


**DISCLAIMER: I/we DO NOT claim any rights over the video obtained by running inference on an OFFICIAL BundeSliga Football Match. We simply use it for Hackathon / Learning purposes and nothing else. Please contact adityaladwa11@gmail.com in case of any issue.**

# Output Inference Preview

![Inference Preview](output_vid3.gif)

[▶️ Full MP4 Video](https://raw.githubusercontent.com/aditya-ladawa/brand_presence_prediction/main/output_vid3.mp4)

# Brand names:
```
adidas
albatros
allianz
american express
antom
aok
apotalde
audi
avaay
balnovade
barmenia
barmenia gothaer
betano
bitburger
bitpanda
blaklader
comedycentral
condor
contienentals
deutschebank
dv system
dvag
dws
ebay
edeka
emirates
ergo
flyeralarm
freshfoodz
gebhardt
gridstore
hormann
hyundai
kempf led
kermi
kleber
lidl
lotto
mainova
mewa
new balance
niedex
paulner
pepsi
prezero
puma
raisin
reuter
rewe
rinti
rmv
robomarket
samsung
santander
sap
sauter
smartbroker
sonepar
sovanta
sparkasse
sportwetten
t
viessmann
volksbank
vonovia
vw
wiesenhof
wurth
xmcyber
yorem
zalando
```

# Brand Exposure Report

- Video: `input_video.mp4`
- Duration: 2.77 minutes (166.27 seconds)
- FPS: 28.71
- Total frames: 4773

## Top 15 brands by total visible duration

| Rank | Brand | Total duration (s) | Occurrences | Avg duration / occurrence (s) | Occurrences / minute | Avg relative area (visible) | Position zone |
|------|-------|--------------------|-------------|--------------------------------|----------------------|-----------------------------|---------------|
| 1 | allianz | 68.94 | 107 | 0.64 | 38.61 | 0.14% | middle-right |
| 2 | t | 59.43 | 137 | 0.43 | 49.44 | 0.60% | bottom-right |
| 3 | audi | 58.07 | 96 | 0.60 | 34.64 | 0.22% | middle-right |
| 4 | barmenia gothaer | 24.35 | 77 | 0.32 | 27.79 | 0.13% | middle-center |
| 5 | adidas | 22.75 | 80 | 0.28 | 28.87 | 0.43% | middle-center |
| 6 | bitburger | 15.78 | 72 | 0.22 | 25.98 | 0.19% | middle-right |
| 7 | edeka | 5.68 | 24 | 0.24 | 8.66 | 0.66% | bottom-center |
| 8 | volksbank | 5.54 | 19 | 0.29 | 6.86 | 0.11% | middle-center |
| 9 | bitpanda | 4.77 | 13 | 0.37 | 4.69 | 0.28% | middle-right |
| 10 | emirates | 4.21 | 20 | 0.21 | 7.22 | 0.22% | middle-center |
| 11 | paulner | 2.65 | 1 | 2.65 | 0.36 | 0.49% | middle-center |
| 12 | niedex | 0.63 | 2 | 0.31 | 0.72 | 0.42% | middle-center |
| 13 | new balance | 0.45 | 5 | 0.09 | 1.80 | 0.37% | bottom-center |
| 14 | kermi | 0.38 | 8 | 0.05 | 2.89 | 0.29% | middle-right |
| 15 | rmv | 0.35 | 5 | 0.07 | 1.80 | 0.19% | top-right |

## All detected brands (sorted by total duration)

| Brand | Total duration (s) | Frames visible | Occurrences | Avg duration / occurrence (s) | Occurrences / minute |
|-------|--------------------|----------------|-------------|--------------------------------|----------------------|
| allianz | 68.94 | 1979 | 107 | 0.64 | 38.61 |
| t | 59.43 | 1706 | 137 | 0.43 | 49.44 |
| audi | 58.07 | 1667 | 96 | 0.60 | 34.64 |
| barmenia gothaer | 24.35 | 699 | 77 | 0.32 | 27.79 |
| adidas | 22.75 | 653 | 80 | 0.28 | 28.87 |
| bitburger | 15.78 | 453 | 72 | 0.22 | 25.98 |
| edeka | 5.68 | 163 | 24 | 0.24 | 8.66 |
| volksbank | 5.54 | 159 | 19 | 0.29 | 6.86 |
| bitpanda | 4.77 | 137 | 13 | 0.37 | 4.69 |
| emirates | 4.21 | 121 | 20 | 0.21 | 7.22 |
| paulner | 2.65 | 76 | 1 | 2.65 | 0.36 |
| niedex | 0.63 | 18 | 2 | 0.31 | 0.72 |
| new balance | 0.45 | 13 | 5 | 0.09 | 1.80 |
| kermi | 0.38 | 11 | 8 | 0.05 | 2.89 |
| rmv | 0.35 | 10 | 5 | 0.07 | 1.80 |
| dvag | 0.03 | 1 | 1 | 0.03 | 0.36 |
