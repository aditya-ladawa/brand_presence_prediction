# Finetuned YOLO v11L on a Custom Annotated Dataset of 71 Brand Logos

Project for KI Sports Hackathon

**DISCLAIMER:**
This project uses broadcast Bundesliga footage purely for hackathon and learning purposes.
No rights are claimed over the source video.
For any concerns, contact **[adityaladwa11@gmail.com](mailto:adityaladwa11@gmail.com)**.

---

## Dataset, Model Weights and Outputs

All data, annotated images, trained model `.pt` files, inference outputs and the problem statement can be found here:
**[https://drive.google.com/drive/folders/1nffUXhxrSZRYhxfjrMXDjRzJDQPFf39a?usp=sharing](https://drive.google.com/drive/folders/1nffUXhxrSZRYhxfjrMXDjRzJDQPFf39a?usp=sharing)**

---

## Inference Preview

**Full MP4 video:**
[https://raw.githubusercontent.com/aditya-ladawa/brand_presence_prediction/main/output_vid3.mp4](https://raw.githubusercontent.com/aditya-ladawa/brand_presence_prediction/main/output_vid3.mp4)


## Validation Images

Below are sample validation frames generated during inference on the validation split.

**Validation frame 1**
![Validation frame 1](https://raw.githubusercontent.com/aditya-ladawa/brand_presence_prediction/main/inference_results/val_output/1.jpeg)

**Validation frame 2**
![Validation frame 2](https://raw.githubusercontent.com/aditya-ladawa/brand_presence_prediction/main/inference_results/val_output/2.jpeg)

**Validation frame 3**
![Validation frame 3](https://raw.githubusercontent.com/aditya-ladawa/brand_presence_prediction/main/inference_results/val_output/3.jpeg)

**Validation frame 4**
![Validation frame 4](https://raw.githubusercontent.com/aditya-ladawa/brand_presence_prediction/main/inference_results/val_output/4.jpeg)

**Validation frame 5**
![Validation frame 5](https://raw.githubusercontent.com/aditya-ladawa/brand_presence_prediction/main/inference_results/val_output/5.jpeg)

---

# Target Brand Classes (71 logos)

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

---

# Brand Exposure Report from Inference

* **Video:** input_video.mp4
* **Duration:** 2.77 minutes (166.27 seconds)
* **FPS:** 28.71
* **Total frames:** 4773

## Top 15 Brands by Total Visible Duration

| Rank | Brand            | Total duration (s) | Occurrences | Avg duration/occ (s) | Occurrences/min | Avg visible area | Position zone |
| ---- | ---------------- | ------------------ | ----------- | -------------------- | --------------- | ---------------- | ------------- |
| 1    | allianz          | 68.94              | 107         | 0.64                 | 38.61           | 0.14%            | middle-right  |
| 2    | t                | 59.43              | 137         | 0.43                 | 49.44           | 0.60%            | bottom-right  |
| 3    | audi             | 58.07              | 96          | 0.60                 | 34.64           | 0.22%            | middle-right  |
| 4    | barmenia gothaer | 24.35              | 77          | 0.32                 | 27.79           | 0.13%            | middle-center |
| 5    | adidas           | 22.75              | 80          | 0.28                 | 28.87           | 0.43%            | middle-center |
| 6    | bitburger        | 15.78              | 72          | 0.22                 | 25.98           | 0.19%            | middle-right  |
| 7    | edeka            | 5.68               | 24          | 0.24                 | 8.66            | 0.66%            | bottom-center |
| 8    | volksbank        | 5.54               | 19          | 0.29                 | 6.86            | 0.11%            | middle-center |
| 9    | bitpanda         | 4.77               | 13          | 0.37                 | 4.69            | 0.28%            | middle-right  |
| 10   | emirates         | 4.21               | 20          | 0.21                 | 7.22            | 0.22%            | middle-center |
| 11   | paulner          | 2.65               | 1           | 2.65                 | 0.36            | 0.49%            | middle-center |
| 12   | niedex           | 0.63               | 2           | 0.31                 | 0.72            | 0.42%            | middle-center |
| 13   | new balance      | 0.45               | 5           | 0.09                 | 1.80            | 0.37%            | bottom-center |
| 14   | kermi            | 0.38               | 8           | 0.05                 | 2.89            | 0.29%            | middle-right  |
| 15   | rmv              | 0.35               | 5           | 0.07                 | 1.80            | 0.19%            | top-right     |

---

# All Detected Brands (Sorted by Total Duration)

| Brand            | Duration (s) | Frames | Occurrences | Avg/occ (s) | Occurrences/min |
| ---------------- | ------------ | ------ | ----------- | ----------- | --------------- |
| allianz          | 68.94        | 1979   | 107         | 0.64        | 38.61           |
| t                | 59.43        | 1706   | 137         | 0.43        | 49.44           |
| audi             | 58.07        | 1667   | 96          | 0.60        | 34.64           |
| barmenia gothaer | 24.35        | 699    | 77          | 0.32        | 27.79           |
| adidas           | 22.75        | 653    | 80          | 0.28        | 28.87           |
| bitburger        | 15.78        | 453    | 72          | 0.22        | 25.98           |
| edeka            | 5.68         | 163    | 24          | 0.24        | 8.66            |
| volksbank        | 5.54         | 159    | 19          | 0.29        | 6.86            |
| bitpanda         | 4.77         | 137    | 13          | 0.37        | 4.69            |
| emirates         | 4.21         | 121    | 20          | 0.21        | 7.22            |
| paulner          | 2.65         | 76     | 1           | 2.65        | 0.36            |
| niedex           | 0.63         | 18     | 2           | 0.31        | 0.72            |
| new balance      | 0.45         | 13     | 5           | 0.09        | 1.80            |
| kermi            | 0.38         | 11     | 8           | 0.05        | 2.89            |
| rmv              | 0.35         | 10     | 5           | 0.07        | 1.80            |
| dvag             | 0.03         | 1      | 1           | 0.03        | 0.36            |

---

# The training results:

 - Dataset is heavily imbalanced and logos vary a lot in position and are mostly very small.
 - Losses decrease smoothly and metrics stabilize, meaning training was healthy, but final recall and mAP values indicate the dataset complexity limits performance.
 - Recall is highest at 0 confidence, around 0.68, and drops sharply, showing the model misses many objects when filtering out low-confidence outputs.
 - mAP@0.5 is 0.492, which is average for a 71-class logo dataset and indicates partial detection reliability.
 - Precision becomes extremely high (0.96) only at confidence near 1.0, meaning the model is accurate when it is sure. Hence we frequently get high-confidence detections.

---

# Training Visualizations

### Box F1 Curve

![BoxF1](https://raw.githubusercontent.com/aditya-ladawa/brand_presence_prediction/main/inference_results/BoxF1_curve.png)

### Box Precision

![BoxP](https://raw.githubusercontent.com/aditya-ladawa/brand_presence_prediction/main/inference_results/BoxP_curve.png)

### Box Recall

![BoxR](https://raw.githubusercontent.com/aditya-ladawa/brand_presence_prediction/main/inference_results/BoxR_curve.png)

### Box Precision-Recall

![BoxPR](https://raw.githubusercontent.com/aditya-ladawa/brand_presence_prediction/main/inference_results/BoxPR_curve.png)

### Labels Distribution

![Labels](https://raw.githubusercontent.com/aditya-ladawa/brand_presence_prediction/main/inference_results/labels.jpg)

### Training Curves

![Results](https://raw.githubusercontent.com/aditya-ladawa/brand_presence_prediction/main/inference_results/results.png)
