### Computer Vision
# Multiple Object Tracking
- This project of [Jupyter Notebook](https://jupyter.org/) by [Python](https://www.python.org/) learn show how to create a complete system of **Computer Vision**

### 1. Team Information
|No.| Members          | Email                         | Role                 |
|---|------------------|-------------------------------|----------------------|
| 1 | Tri Nguyen Phan  | 20127578@student.hcmus.edu.vn | Team Leader          |
| 2 | Manh Hung Nguyen | 20127030@student.hcmus.edu.vn | Algorithm Developer  |
| 3 | Hoang Minh Luu   | 20127048@student.hcmus.edu.vn | UI/UX Designer       |
| 4 | Hoang Kim Tran   | 19127039@student.hcmus.edu.vn | Algorithm Programmer |

### 2. Overview
- Support Software: [Visual Studio Code](https://code.visualstudio.com/), [Google Colabotary](https://colab.google/), [Github](https://github.com/).
- Programming Languages: [Python](https://www.python.org/)
- In this modern age, Object Tracking has a wide range of applicational significances in various fields, including the applicational significance of object tracking enables real-time monitoring, analysis, and control of various processes and systems. 
- Lying on image processing, the topic of Object tracking is a process of locating and following a specific object or region of interest (ROI) in a video stream or a sequence of images.
- Object Tracking is the automatic identification and tracking of objects in videos or image sequences, with applications in computer vision, machine learning, robotics, and augmented reality. It can be used for surveillance, traffic monitoring, sports analysis, and studying behaviour patterns.
- Basic understanding of the system
    - Input of the system: a sequence of frames from a video containing one or more objects.
    - Output of the system: Trackless which includes the object’s frame, bounding boxes, and identification over time.

|Input|Output|
|-------------------------------------------|-------------------------------------------|
|<img src="https://i.imgur.com/Gbb1isv.png">|<img src="https://i.imgur.com/KDIqiZ6.png">|

- Multiple Object Tracking (MOT) is the process of simultaneously tracking multiple objects in a video sequence or a set of images. MOT is a challenging problem in computer vision and has many practical applications, including traffic monitoring, crowd analysis, and robotics.
- The goal of MOT is to assign a unique identity to each object and estimate their locations and trajectories over time. MOT algorithms typically use a combination of object detection, data association, and motion estimation techniques to track multiple objects.

### 3. Contributions
|Solutions|Detail Contributions|
|---------|--------------------|
|Online Tracking|Using only the current frame and the immediately preceding frame for tracking. May reduce the accuracy of the algorithm, but it reflects the practical online nature of the problem|
|Offline Tracking|Typically use the entire video frames, thus achieving much higher accuracy compared to Online Tracking|
|Detection-based Tracking (2-stage)|Focusing on the close relationship between **Object Detection** and **Object Tracking**. Using the results of detection to track the object across frames|
|Summary of Multi-Object Tracking|Tracking by Detection (2-stage): Step 1: Detector for object detection: *YOLO*; Step 2: Object association among frames: Appearance matching, Motion analysis. Detection Free Tracking (1-stage): Detection and tracking steps are simultaneously produced in a single network|

### 4. Methodology
#### 4.1 OBJECT DETECTION
- Input: Single image
- Output: Region proposals which include bounding boxes and feature maps (by Roi Align)

#### 4.2 QUASI-DENSE SIMILARITY LEARNING
<img src="https://i.imgur.com/FiEc7YV.png">

- Input: Keyframe, reference frame.
- Output: each frame has its region proposals which include bounding boxes and feature embeddings (feature maps of RoIs are extracted by an embedding head).

#### 4.3 OBJECT ASSOCIATION
<img src="https://i.imgur.com/CDtlkKX.png">

- Input: Keyframe, reference frames.
- Output: Track identification for each object (after matching the successful embedding feature by “bi-softmax”) in the keyframe. 

### 5. Pseudocodes
```sh
Class Embedding_Extractor:
	Function init(config):
		self.backbone = create _backbone(config.backbone_name)
		self.rpn = create_region_proposal_network(config.rpn)
		self.box_head = create_convolution2D(config.box_head)
		self.embedding_head = create_convolution2D(config.embed_head)
	Function forward(single_frame):
		visual_features = self.backbone(single_frame)
		RoIs_features = self.rpn(self.visual_features)
		RoIs_boxes = self.box_head(self.visual_features)
		RoIs_features = self.embedding_head(RoIs_features)
		Return {“boxes”: RoIs_boxes, “features”: RoIs_features}
```

```sh
Class QDTrack:
	Function init(config):
		self.extractor = Embedding_Extractor(config)
		self.tracklets = []
		# select similarity function (”Bi-directional softmax” or “cosine")
		self.similariy_function = select_similarity_function(default=”Bi-directional softmax”) 
	Function forward(single_frame):
		RoIs = self.extractor(single_frame)
		RoIs_features = RoIs[“features”]
		Tracklets_features = self.tracklets[“features”]
		Similarity_scores = self.similariy_function(RoIs_features, Tracklets_features)
		Assigned_RoIs = Assign_ID(RoIs, self.tracklets, Similariy_scores, config.threshold)
		Update Assigned_RoIs to self.tracklets
		Return self.tracklets
```

```sh
Main_function(config, video):
    model = QDTrack(config)
    track_result = [] # [frame_id, track_id, x_left, y_left, width, height, 1, 1, 1]
    
    For each frame in sequence of video:
    	Result = model(frame)
    	Record_and_show_result(track_result, Result)
    Save track_result to file
```

### 6. Demonstration
- The DanceTrack dataset is a video dataset of multiple people dancing in a crowded environment, where occlusions and interactions among the dancers are common.
- In this project, our team would take up to 25 videos in a validated set with the length of each video from 80 seconds to 160 seconds. The dataset provides ground truth annotations for the positions and identities of the dancers in each frame.
- Our team has proposed a demonstration of how to run a source code which applies Quasi-dense Similarity Learning for the Multiple Object Tracking method on [the DanceTrack dataset](https://github.com/DanceTrack/DanceTrack) as [the following demo video](https://drive.google.com/file/d/1jZlghnBrB3cLdwpJ9VEvXgVwVBgJHLk5/view)


### 7. In Conclusion
- **QDTrack**, employing a unique quasi-dense similarity learning framework, excels in multiple object tracking, particularly in challenging scenarios like crowded scenes with occlusions. Compared to leading methods, QDTrack demonstrates competitive performance on key metrics. It adeptly handles issues such as long-term occlusions and track fragmentation. However, its suitability depends on the presence of object proposals and assumptions of stable object appearance and motion patterns. In scenarios deviating from these conditions, alternative tracking methods may be more suitable.

### 8. Further Information
- For further details, please visit the **Report.pdf** file provided above.
