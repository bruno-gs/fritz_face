<h1 align="center">Fritz Recognizer - Images</h1>

## :star: Motivation

This work was done in 2020, during the quarantine (Covid-19). In this scenario, the CBR (Brazilian Robotics Competition) took place in a virtual environment in November 2020. 

One of the tasks to be developed by the teams was the identification of People in an image, drawn at each round and each team. 

The objective was to identify if the people in the image were wearing a mask or not. If they were without a mask, to extract the gender and age of each one.

Representing the RoboFEI@HOME team, I used the works referenced in the file "link.txt" and combined them to achieve the objective.
 
## :hourglass: CBR2020 Virtual - @Home - Exam: People Recognition
:movie_camera: [YouTube Demo Link](https://www.youtube.com/watch?v=EU1RUpT1pf0&t=2920s)

Our results can be seen in the video time:

- 47:00
- 1:22:50
- 2:01:25


## :warning: TechStack/framework used

- [OpenCV](https://opencv.org/)
- [Caffe-based face detector](https://caffe.berkeleyvision.org/)
- [Keras](https://keras.io/)
- [TensorFlow](https://www.tensorflow.org/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)

## :key: Prerequisites

All the dependencies and required libraries are included in the file <code>requirements.txt</code> [See here](https://github.com/bruno-gs/fritz_recognizer_image/blob/main/requirements.txt)


## 🚀&nbsp; Installation
1. Clone the repo
```
$ git clone https://github.com/bruno-gs/fritz_recognizer_image
```

2. Now, run the following command in your Terminal/Command Prompt to install the libraries required
```
$ pip3 install -r requirements.txt
```

## :bulb: Working

1. Open terminal. Go into the cloned project directory and type the following command:
```
$ python3 train_mask_detector.py --dataset dataset
```

2. To detect face masks in an image type the following command: 
```
$ python3 detect_mask_image.py --image images/pic1.jpeg
```

3. To detect face masks in real-time video streams type the following command:
```
$ python3 detect_mask_video.py 
```
## :clap: And it's done!
Feel free to mail me for any doubts/query 
:email: chandrikadeb7@gmail.com

## :handshake: Contribution
Feel free to **file a new issue** with a respective title and description on the the [Face-Mask-Detection](https://github.com/chandrikadeb7/Face-Mask-Detection/issues) repository. If you already found a solution to your problem, **I would love to review your pull request**! 

## :heart: Owner
Made  by [Bruno Gottsfritz](https://github.com/bruno-gs)

## :eyes: License
MIT © [Bruno Gottsfritz](https://github.com/bruno-gs/fritz_recognizer_image/blob/main/LICENSE)
