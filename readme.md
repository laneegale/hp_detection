# Catching Colon Bacteria on the beachheads

Workflow
patient stomache, gets endoscopy, extract tissue by endoscope
if bacteria is detected, prescribe antibiotic
most of the time, positive and negataive groups are well separated 
marginal case requires pathologist to look around the beaches and check for bacteria
physicallty traverse the slide once
this is ripe for autoamtion

1. Distinguish beaches
2. Detect bacteria on beaches


## Data

### Summary
tile size 512x512
level = 0
N slides = 5
N tiles : 
0 : 351
1 : 502



### Raw Data

- excel (HPACG+ or HPACG- labels)

    `X:\KAY\Datasets\Stomach biopsy`

    - Feature column : HPACG
    - blank : no/mild gastritis N=2798-237
    - 1 : postive (target class) N=237
    - no file for some cases


- slides

    `X:\KAY\Datasets\Stomach biopsy\Stomach Biopsy(svs)1` - 66 HPACG+

    `X:\KAY\Datasets\Stomach biopsy\Stomach Biopsy(mrxs)1`


> there are more tasks in the excel but it would require even more annotation at different field of view.

> prime said he will look into using GRAD CAM heatmap -? generate bbox -> box as prompt into SAM

## Method

### Tissue Boundary Detection
Works reasonably well. See the labelling ROI boxes.

### Labelling

Tissue area is segmented. Shoreline is derived from the tissue area. Bounding boxes are created in geojson by `sliderdicer.py`. Geojsons are imported to Qupath by script `import_geojson_2025.groovy`.

All boxes have a default class of HPACG-. To lighten labelling load.
First 5 images, where the slides are .svs and are HPACG+ in the excel, are used as a pilot. 

All tools are currently developed for .svs. .mrxs support requires further work.


<!-- ![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-3.png)
![alt text](image-4.png)
![alt text](image-5.png) -->


<img src="image-2.png" height="150"/> 
<img src="image-3.png" height="150"/> 
<img src="image-4.png" height="150"/> 
<img src="image-5.png" height="150"/> 
<img src="image.png" height="150"/>

<img src="image-6.png" height="256"/>
<img src="image.png" height="256"/>
<!-- ![alt text](image-6.png)
![alt text](image.png) -->

LEGEND : <span style="color: red;">HPACG-</span>, <span style="color: green;">HPACG+</span>, <span style="color: orange;">Other, not trained</span>.

### Example tile
### Class 0
![alt text](image-10.png)
### Class 1
![alt text](image-12.png)


#### Difficult tiles
HPACG+
<img src="image-7.png" height="256"/>
<img src="image-8.png" height="256"/>



#### Other methods
using immune cells to assist, requires label
![alt text](image-9.png)


### Training
epoch = 500
input shape = 512x512x3
efficienetb3
training time : 6.5 hours


### Evalution 
pertty terrible
![alt text](image-14.png)
![alt text](image-13.png)

### UNI+Logistic Regression
train : test = 7:3

![alt text](/mnt/nas204/Prime_D/HPACG_classiciation/trainsetting.png)
![alt text](/mnt/nas204/Prime_D/HPACG_classiciation/matrix.png)
![alt text](/mnt/nas204/Prime_D/HPACG_classiciation/AUC.png)
