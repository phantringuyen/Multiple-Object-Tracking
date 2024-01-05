### Computer Vision
# Multiple-Object-Tracking
- This project of [Jupyter Notebook](https://jupyter.org/) by [Python](https://www.python.org/) learn show how to create a complete system of **Computer Vision**

### 1. Team Information
|No.| Members          | Email                         | Role                 |
|---|------------------|-------------------------------|----------------------|
| 1 | Tri Nguyen Phan  | 20127578@student.hcmus.edu.vn | Team Leader          |
| 2 | Manh Hung Nguyen | 20127030@student.hcmus.edu.vn | Algorithm Developer  |
| 3 | Hoang Minh Luu   | 20127048@student.hcmus.edu.vn | UI/UX Designer       |
| 4 | Hoang Kim Tran   | 19127039@student.hcmus.edu.vn | Algorithm Programmer |

### 2. Overview
- Support Software: Visual Studio Code, Google Colabotary, Github.
- Programming Languages: Python
- In this modern age, Object Tracking has a wide range of applicational significances in various fields, including the applicational significance of object tracking is that it enables real-time monitoring, analysis, and control of various processes and systems. 
- Lying on image processing, the topic of Object tracking is a process of locating and following a specific object or region of interest (ROI) in a video stream or a sequence of images.
- Object Tracking is the automatic identification and tracking of objects in videos or image sequences, with applications in computer vision, machine learning, robotics, and augmented reality. It can be used for surveillance, traffic monitoring, sports analysis, and studying behavior patterns.
- Basic understanding of the system
    - Input of the system: a sequence of frames from a video contains one or more objects.
    - Output of the system: Trackless which includes the object’s frame, bounding boxes, identification over time.

|Input|Output|
|-------------------------------------------|-------------------------------------------|
|<img src="https://i.imgur.com/Gbb1isv.png">|<img src="https://i.imgur.com/KDIqiZ6.png">|


- Each image has a size of 28x28 pixels. The pixel values range from 0 to 255, representing the brightness or darkness of each pixel. The training dataset (.csv) consists of 785 columns. The first column represents the label, indicating which digit is depicted, and the remaining 784 values represent the pixel values of the image.

<img src="https://media.licdn.com/dms/image/C5112AQHIlfAil0puPQ/article-inline_image-shrink_1000_1488/0/1576237874564?e=1709769600&v=beta&t=SjVGbWIES8fsgInGsnpify9F8-oBtSODQzT-p93KTTo">

- It is evident from the illustrated example that although there are only ten data points (representing the ten handwritten digits from 0 to 9), each point has 28x28 = 785 pixels. In situations like this, constructing a model that captures all features to understand the essence of the data is infeasible. To overcome this, efforts are made to increase the number of data points or reduce the dimensionality of each point.
- In this context, the research will delve into Principal Component Analysis (PCA) as a dimensionality reduction method. PCA is a technique that transforms a high-dimensional dataset (a dataset with many features/variables) into a low-dimensional dataset.
- Given the provided statistics, when applying PCA to reduce the dimensionality to only two, we obtain the following graph:

<img src="https://i.imgur.com/TLwhbSk.png">

- With the graph above, the data has been transformed from 785 dimensions down to 2 dimensions. The data points of label 0.0 (dark blue points) exhibit significant differences compared to other labels. Understanding the nature of this data aids in making informed decisions. In this example, the handwritten digit images of label 0.0 have some unique characteristics compared to other data points.
- This serves as a fundamental example of the utility of PCA in data exploration, highlighting its ability to reduce dimensionality while retaining the most important features. This not only enhances the efficiency of data analysis but also simplifies the process of visualizing and interpreting the data.

### 3. General Framework:
|Phase| Framework                                                                             |
|---|-----------------------------------------------------------------------------------------|
| 1 |Import the necessary libraries to apply PCA algorithms to the provided data|
| 2 |Read data from the provided CSV file|
| 3 |Specify the required input parameters to run the program|
| 4 |Write individual functions to perform some basic multivariate analysis with visualization|
| 5 |Load the input parameters into PCA functions and print the results (a plot)|

### 4. Useful of PCA

|No.| Uses                                                                             |
|---|-----------------------------------------------------------------------------------------|
| 1 |Extract the most important information from the data set|
| 2 |Reduce data dimensionality with the ability to minimize information loss and maximize data variability|
| 3 |Solve the problem and analyze the correlation between variables in the original data using the new variables that the algorithm has found|

### 5. Team Contributions
- Firstly, the members of the team provided a clear explanation and a scientific formula for the Principal Component Analysis (PCA) problem, starting from its definition and purpose, and outlining the steps involved in its implementation. This explanation will be beneficial for anyone seeking to understand PCA, regardless of their foundational knowledge.
- Additionally, the application of PCA to a real-world problem, specifically the classification of the IRIS dataset, was discussed. This application is highly valuable in demonstrating how PCA can be utilized in practical scenarios to reduce the dimensionality of the dataset and improve classification accuracy.
- Last but not least, the team also presented visual aids to illustrate PCA concepts, including scatter plots of the original and transformed datasets, and an explained cumulative variance plot. These visuals are instrumental in comprehending the impact of PCA on the dataset and assessing the quality of PCA results.

### 6. User guide
- With the program that the team has developed, the source code has been created and tested based on the detailed description of the Iris flower dataset. To obtain this dataset, users can search on the internet as it is an open-source dataset, or the team may have already prepared this dataset.
- The team has also prepared a video recording to document the process of downloading the experimental dataset and the source code of this program, aiming to facilitate users in accessing and running the team's source code. Here is the link to the [team's demo video](https://drive.google.com/file/d/1wE0kdBCmJIuVXl3CZ5yajF119s6HVG34/view?usp=sharing).
- The following are detailed steps providing instructions on how to use the team's Principal Component Analysis project source code.
  - Step 1: We utilized the Google Colaboratory browser to run the PCA source code. Below are the links to access the PCA source code and the Iris.csv dataset. Users can download the source code and the experimental dataset for PCA:
    - [PCA Source Code](https://colab.research.google.com/drive/1kz-Gj7uqvpAfeq5lVlnVr2E6t-OgOUIR)
    - [Iris.csv Dataset](https://drive.google.com/file/d/1odvlEvwqIii8DDYdROKefBD_k0ZN9g8b/view?usp=sharing)

<img src="https://i.imgur.com/E7frp7I.png">

  - Step 2: Once you have obtained the PCA source code and the Iris.csv dataset, users should upload the source code onto Google Colaboratory to run the program. Open the Google Colaboratory in your web browser. Next, click on "File" in the top-left corner of the Colab interface. After that select "Upload" from the dropdown menu. Then Choose the PCA source code file that you have downloaded to upload it to Google Colab.

<img src="https://i.imgur.com/PUUQJBp.png">

  - Next, click on the Choose File frame, a File Folder will appear, the user selects the file Seminar02_Group10_PTTKDLNB_PCA which is the source code of the group, select Open and wait for Google Colab to open the source code. If the user screen displays the source code as shown below, it means the user has successfully opened the group's program.

<img src="https://i.imgur.com/572ZOvG.png">


  - Step 3: Following that, users can choose the default hardware or GPU because the program we have developed does not consume too much processing memory resources, here we will demo the default hardware format of the device. Google Colab to not consume significant space.

<img src="https://i.imgur.com/y2oK7B7.png">

  - Step 4: In step 4, users can run each line of code in the program or select **Runtime** => **Run** all to run the entire program. One thing to note is that because the program needs an experimental data set, the dataset will be downloaded from the GitHub link the group posted before, and then users will be asked to allow Google Colab to link to Google Drive to download the data set. Data for the experimental program of Principal Component Analysis algorithm.

  - Step 5: When the system runs, the final result like the demo screen shows that the user has successfully executed the PCA algorithm demo program.

<img src="https://i.imgur.com/NhvvukM.png">



### 7. In Conclusion
|A graph with three distinct components       |A graph with only two distinct components after being applied PCA|
|---------------------------------------------|-----------------------------------------------------------------|
| <img src="https://i.imgur.com/LKmV92M.png"> |<img src="https://i.imgur.com/5tgOdpB.png">                      |

- Based on the chart presented in the table above, we can observe a clear separation among the three species of Iris: Setosa, Versicolor, and Virginica. Setosa stands out distinctly from the other two species, forming a distinct cluster in the bottom-left corner of the plot. On the other hand, data points for the Versicolor and Virginica flowers are somewhat overlapping, yet still forming two separate clusters in the top-right corner of the plot. This may lead to the conclusion that Versicolor and Virginica share similarities in appearance (sepals and petals), while Setosa exhibits relatively distinct features in appearance compared to the other two species.
- Furthermore, the chart indicates that the first principal component (PC1) is more crucial than the second principal component (PC2) in distinguishing Iris species. PC1 captures approximately 92% of the total variability in the data, while PC2 accounts for only about 6%. This suggests that the majority of information in the data can be explained by PC1.
- In general, the PCA algorithm effectively reduced the dimensionality of the Iris dataset while retaining the most important features, enabling more efficient data analysis. The resulting plot clearly illustrates the distribution and separation of different Iris species in two-dimensional space.
- Regarding the runtime of the PCA algorithm, supported by libraries such as Numpy, Pandas, Seaborn, and Matplotlib, the program executed at a very efficient speed.

### 8. Further Information
- For further details, please visit the **Report.pdf** file provided above. Thank you very much and Have a nice day!
