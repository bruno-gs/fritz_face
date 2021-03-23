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

Or in the "log" folder, there is the result of the image already contained in the "images" folder

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

1. Place the image you wish to analyze in the "images" folder.

2. Open terminal. To detect face masks or age and gender in an image type the following command:
```
$ python3 mask_age_gender.py --image images/name_of_image.jpeg
```
3. The result will appear on your screen, but this log will also be in the "log" folder.

4. If you have any problems and the program doesn't run, it must be because of a missing package, post the error on the Internet. If nothing works, create a issue.

## :clap: And it's done!
Feel free to mail me for any doubts/query 
:email: brugottsfritz@gmail.com

## :heart: Owner
Made  by [Bruno Gottsfritz](https://github.com/bruno-gs)

## :eyes: License
MIT © [License](https://github.com/bruno-gs/fritz_recognizer_image/blob/main/LICENSE)
