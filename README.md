# Juice_CapstoneProject
Data Science Capstone Project

## CP-Task 1
### SDG 3 Ensure healthy lives and promote well-being for all at all ages 
![](/images/heartdisease_mdata.png)

Retrived from [DOSM Malaysia](https://www.dosm.gov.my/v1/index.php?r=column/cthemeByCat&cat=401&bul_id=QTU5T0dKQ1g4MHYxd3ZpMzhEMzdRdz09&menu_id=L0pheU43NWJwRWVSZklWdzQ4TlhUUT09)

Heart disease is a fatal human disease, rapidly increases globally in both developed and undeveloped countries and consequently, causes death. According to the World Health Organization(WHO), heart disease is the number 1 cause of death globally, taking an estimated 17.9 million lives each year, representing 31% of all global deaths. Ischaemic heart disease is the top cause of death in Malaysia, ischaemic heart disease was also recorded as the principal causes of death in Australia(11.1%) and United States(23.1%)(FMT 2020). The most common symptoms of heart disease include physical body weakness, shortness of breath, feet swollen, and weariness with associated signs, etc. The risk of heart disease may be increased by the lifestyle of a person like smoking, unhealthy diet, high cholesterol level, high blood pressure, deficiency of exercise and fitness, etc. Thus, our team decides to work on this topic and determine the knowledge, patterns and relationship associated with heart disease using the heart disease dataset which is retrieved from the internet. By the aid of Machine Learning, we can create a model to do the early detection and treatment of heart disease which will possibly reduce the death rate of heart disease. 

## Goals

1) Reduce medical test costs  

2) Early detection and treatment of heart disease 

3) Reduce death rate of heart disease 

 

## Objectives

1) From the data, we want to find the relationship between the medical factors related to heart disease. 

2) Has a higher chance to intervene much earlier and head off hospitalizations 

3) To create a model to predict heart disease. 

 

## Questions 

1) Does the prediction of heart disease improve the health care system? 

2) Does age affect the risk of getting heart disease? 

3) Does gender affect the risk of getting heart disease? 

 

## Success and its measurement

1) We will determine the confusion matrix and want to have a good prediction in the performance measures such as accuracy, precision and recall. 

2) The model can apply on the unseen test data on real world and predict accurately.  


## Data sources 

[Heart Disease data source](https://archive.ics.uci.edu/ml/datasets/heart+disease), our dataset is retrieved from UCI Machine Learning Repository. 

```python
import pandas as pd
data = pd.read_csv('Downloads/cleveland.csv',header=None)
data
```
<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1 0 63 1 -9 -9 -9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-9 1 145 1 233 -9 50 20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1 -9 1 2 2 3 81 0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0 0 0 0 1 10.5 6 13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>150 60 190 90 145 85 0 0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2975</th>
      <td>1 888 -9 4016 8216 8216 788 0 -9 -9 -9</td>
    </tr>
    <tr>
      <th>2976</th>
      <td>-0 0 0 1 9 -9 130 80 0 130 80 0 11</td>
    </tr>
    <tr>
      <th>2977</th>
      <td>-9 3 1h9 1 -9 -9 -9</td>
    </tr>
    <tr>
      <th>2978</th>
      <td>-9 3 -9 -9 -9</td>
    </tr>
    <tr>
      <th>2979</th>
      <td>-9 -9 -9 3 -9 -4 1 1</td>
    </tr>
  </tbody>
</table>
<p>2980 rows × 1 columns</p>
</div>


<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1 0 63 1 -9 -9 -9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-9 1 145 1 233 -9 50 20</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1 -9 1 2 2 3 81 0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0 0 0 0 1 10.5 6 13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>150 60 190 90 145 85 0 0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>2975</th>
      <td>1 888 -9 4016 8216 8216 788 0 -9 -9 -9</td>
    </tr>
    <tr>
      <th>2976</th>
      <td>-0 0 0 1 9 -9 130 80 0 130 80 0 11</td>
    </tr>
    <tr>
      <th>2977</th>
      <td>-9 3 1h9 1 -9 -9 -9</td>
    </tr>
    <tr>
      <th>2978</th>
      <td>-9 3 -9 -9 -9</td>
    </tr>
    <tr>
      <th>2979</th>
      <td>-9 -9 -9 3 -9 -4 1 1</td>
    </tr>
  </tbody>
</table>
<p>2980 rows × 1 columns</p>
</div></div>


This is the unprocessed dataset of heart disease which contains 2980 rows and 1 column. From the figure above, we can visualize that each row contains different number of features. 

**(Elements 5V)** 

1) **Volume**  

  The dataset we use contains 2980 rows. It can be considered as a large amount of data.  

2) **Velocity** 

  Every year, the heart disease has the highest number of deaths and increases so far resulting in the data increases. 

3) **Variety** 

  The dataset is an unstructured data which is haven’t been processed yet. The dataset also comes from different countries such as United States, Hungary and Switzerland.  

4) **Veracity** 

  The dataset contains large number of data which is sometimes get messy and bad quality. The dataset also contains 75 features which have different dimension. The data in 			     dataset is comes from trustworthy sources which are either from hospital or institute. 

5) **Value** 

  At the beginning, raw data may not helpful to the stakeholder or the related organization. However, after the data has been cleaned and processed, it might bring some useful 	   value and insight about heart disease. Then, we are able create a model to predict heart disease using the interesting information and pattern found. 

## References 

1) [https://www.freemalaysiatoday.com/category/nation/2020/11/26/heart-disease-top-killer-in-malaysia-involving-69-males/](https://www.freemalaysiatoday.com/category/nation/2020/11/26/heart-disease-top-killer-in-malaysia-involving-69-males/)

2) [https://www.who.int/health-topics/cardiovascular-diseases/#tab=tab_1](https://www.freemalaysiatoday.com/category/nation/2020/11/26/heart-disease-top-killer-in-malaysia-involving-69-males/)
