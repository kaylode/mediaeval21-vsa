# Visual Sentiment Analysis: A Natural Disaster Use-case

The increasing popularity of the social networks and users' tendency towards sharing their feelings, expressions and opinions in text, visual and audio content have opened new opportunities and challenges in sentiment analysis.There has been a great deal of work done related to sentimental analysis of textual data but idea of extracting sentiments from the videos and images has not been addressed the same way. The focus of this task is images related to different natural disasters around the world, able to perform a meaningful analysis on this data could be of great societal importance.

## MediaEval 2021

<details>

### Motivation and background
As implied by the popular proverb "a picture is worth a thousand words," visual contents are an effective means to convey not only facts but also cues about sentiments and emotions. Such cues representing the emotions and sentiments of the photographers may trigger similar feelings from the observer and could be of help in understanding visual contents beyond semantic concepts in different application domains, such as education, entertainment, advertisement, and journalism. To this aim, masters of photography have always utilized smart choices, especially in terms of scenes, perspective, angle of shooting, and color filtering, to let the underlying information smoothly flow to the general public. Similarly, every user aiming to increase in popularity on the Internet will utilize the same tricks. However, it is not fully clear how such emotional cues can be evoked by visual contents and more importantly how the sentiments derived from a scene by an automatic algorithm can be expressed. This opens an interesting line of research to interpret emotions and sentiments perceived by users viewing visual contents.

### Dataset Details
The dataset consist of disaster-related images from all over the world. Each image has been manually annotated by five different people with tags related to emotion generated when viewing the image. If the label is selected by majority of the annotators then label is assigned to that image. There were five different type of question asked to annotators to get a more diverse perspective of the type of emotion these images are invoking. The dataset consists of 2,432 Images ([dev-set](https://drive.google.com/file/d/1PszWQ3Y5TWxxCnIhaJaG7EJjG3YlB_Qn/view?usp=sharing)) and 1,199 Images (test-set). 
Devset can be downloaded from [here](https://drive.google.com/file/d/1PszWQ3Y5TWxxCnIhaJaG7EJjG3YlB_Qn/view?usp=sharing).

### Task Description
Disaster-related images are complex and often evoke an emotional response, both good and bad. This task focuses on performing visual sentiment analysis on images collected from disasters across the world. 
<!-- # Here you need a short sentence so that people know that it is the sentiment expressed by the photographer as judged by crowdsourcing workers-->
The images contained in the provided dataset aim to provoke an emotional response through both intentional framining and based on the contents itself.

*Subtask 1: Single-label Image Classification* The first task aims at a single-label image classification task, where the images are arranged in three different classes, namely positive, negative, and neutral with a bias towards the negative samples, due to the topic taken into consideration. 

*Subtask 2: Multi-label Image Classification* This is a multi-label image classification task where the participants will be provided with multi-labeled images. The multi-label classification strategy, which assigns multiple labels to an image, better suits our visual sentiment classification problem and is intended to show the correlation of different sentiments. In this task seven classes, namely joy, sadness, fear, disgust, anger, surprise, and neutral, are covered. 

*Subtask 3: Multi-label Image Classification* The task is also a multi-label, however, a wider range of sentiment classes are covered. Going deeper in the sentiment hierarchy, the complexity of the task increases.  The sentiment categories covered in this task include  anger, anxiety, craving, empathetic pain, fear, horror, joy, relief, sadness, and surprise.  

### Task Schedule
* 15 October (Tentative) : test-set release <!-- # Replace XX with your date. We suggest setting the date in June-July-->
* 5 November: Runs due <!-- # Replace XX with your date. We suggest setting enough time in order to have enough time to assess and return the results by the Results returned deadline-->
* 15 November: Results returned  <!-- Replace XX with your date. Latest possible should be 15 November-->
* 22 November: Working notes paper  <!-- Fixed. Please do not change. Exact date to be decided-->
* 6-8 December: MediaEval 2021 Workshop <!-- Fixed. Please do not change. Exact date to be decided-->
* 
### Evaluation methodology
All the tasks will be evaluated using standard classification metrics, where F1-Score will be used to rank the different submissions. We also encourage participants to carry out a failure analysis of the results to gain insight into why a classifier may make a mistake.
<!-- # This description needs to make clear what the crowdworkers were actually asked. It seems that they are not reporting their own experience of the emotional impact of the photographs, but rather the intention of the photographer-->

### Submission
For this task you may submit up to 2 runs fro each task.

#### Submission Format
Please submit your runs for the task in the form of a csv file, where each line contains one Image ID followed by the label for the T1,T2 and T3. Image IDs and the labels should be comma separated. For reference please follow the devset GT format.

</details>


