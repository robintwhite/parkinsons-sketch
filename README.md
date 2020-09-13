# parkinsons_sketch
 Analysis of sketches from people suffering from Parkinsons <br>
 Classification using 3 different model types: <br>
 * Random Forest using attributes from image processing operations, such as number of intersections and line thickness.
 * Logistic Regression from features obtained through ResNet50 architecture
 * Trained convolutional NN model from scratch. Architecture in line with mini VGG. Conv followed by dense. 
 
<br>
Link to Medium [post](https://medium.com/p/classifying-parkinsons-disease-through-image-analysis-2e7a152fafc9?source=email-c5eb85d3a614--writer.postDistributed&sk=e14a3373451cdf1399605017bf662c05)
<br>
![alloneplot](./images/all_one_plot.png)
<br>
Create conda environment: 
```
conda env create -f parkinsons-sketch.yml
```
