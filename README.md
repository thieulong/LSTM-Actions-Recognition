## To effectively create your custom LSTM model, please follow the steps below:
### Step 0: Install the requirements
In order to run this application, we start with installing the dependencies and libraries using this command:
```
pip install -r requirements.txt
```
Note: It will be better if you create a virtual environment for this to avoid conflicts
### Step 1: Create your dataset
There are 2 files to create your custom dataset (Pose and Hands), named as `pose_data_generation.py` and `hands_data_generation.py`, choose one based on your need, and then you'll need to mofidy the parameters inside the script, details as below:
- Change the image strem index (line 5) according to your device, if you are using a webcam from your laptop, leave it at 0 or 1, if you are using an external USB, you will have to change it according to the index of your device.
- Change the label name (line 12), change it to the label of your dataset, for example, if you're about to train a dataset of push-ups, then change it to `label = "push-ups"`
- Change the number of frame/data you want to train (line 13), currently it is at 1000, sufficient for a small model, if you aim for a more precise model, I'd recommend to increase it to 2000 - 5000
- For `hand_data_generation.py` you can determine how many hands you want to capture the motion (line 8), replace max_num_hands to 2 if you want to capture both hands, leave it at 1 and you'll capture the first hand detected.

Then run the script with:
```
python3 hand_data_generation.py
```
or
```
python3 pose_data_generation.py
```
Note: Once you executed the script, a window with image stream from your device's camera will pop-up, and once you can see the bounding box or your landmarks appear, it means the data is being generated, from this point, start performing the action for your datasets, be careful to always performing the action you want the model to learn, avoid performing other irrelevant actions since it can generate noise in the datasets.

For example, if you're training a push-ups dataset, make sure you get your setup right, have the camera pointing at your entire body, and once you executed the script, start doing some push-ups too 💪💪

Once the script terminated, you'll see a .txt file generated with your label, check if that file has the number of lines approximately match with your number of frame/data declare above (line 13). If it is, consider this step is done, if it doesn't match, there must be some problems and you'll need to execute the script again to regenerate the data.

### Step 2: Train the LSTM model
Once you got the dataset, let's train the model. Start with opening the `train_model.py` file, add or remove the number of datasets you generated before (line 8 - 11). For example, if you generate 3 datasets in total: push-ups, plank and burpees you'll need to declare those 3 datasets like:  
```
push-ups_df = pd.read_csv("push-ups.txt")
plank_df = pd.read_csv("plank.txt")
burpees_df = pd.read_csv("burpees.txt")
```
Note: Remember to delete the default datasets, only declare the exact dataset you generated in Step 1.  
  
Next, you'll need to prepare the data for the model training, which involves categorizing the dataset and split them for training and validation, as below:

Remove line 17 - 39 in the default code based on your dataset, with the example of having 3 datasets above (push-ups, plank, burpees), we will now just have 3 datasets in the code:
```
datasets = push-ups_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(0)

datasets = plank_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(1)  

datasets = burpees_df.iloc[:, 1:].values
n_samples = len(datasets)
for i in range(no_of_timesteps, n_samples):
    X.append(datasets[i-no_of_timesteps:i, :])
    y.append(2) 
```
Note: If you only have 2 datasets, only declare and split 2 datasets of yours, in the example above, we have 3 datasets, and in the default code, we have 4.  
Note: Pay attention to the `y.append()` line, make sure you proper declare the index of the dataset because this will be important for further use.

Optional: You can adjust the train/test ratio in line 44, it is current default at train 80% and test 20%.

Once you done all that, let's execute the script to train the model with:
```
python3 train_model.py
```
Note: It suppose to take a while depending on the size of your datasets.

### Step 3: Run a demo with your model
Congratulations on getting to this step, you're almost finish, this step will involve you to runinng a real time demonstration with the model you trained.  
There will be 2 files depending on your application (Pose and Hands), choose the one that match with what you've been training the model for, they're named as `pose_lstm_realtime.py` and `hand_lstm_realtime.py`. Below will be a few things you'll need to adjust before executing the script.  
- Change the default parameter (line 41), default parameter is the parameter that will be initialized at first, change it to one of your dataset's label.
- Change the neutral label (line 42), the term `neutral_label` in here refer to a dataset that is neutral, for example in the `hand_lstm_realtime.py` script, the neutral state in here is when the hand has not grasped an object, and in `pose_lstm_realtime.py` the neutral state in here is the person not performing any violent act. In short, this is not very important at all, but it's the way that I'm developing this application is to separate an action from another action. For example with the 3 actions we have above: push-ups, plank and burpee, we select plank as our neutral_label, then every time a person is detected as doing plank, the box will be in red and the label will be in red as well, but the detection results doesn't matter.
- detect function: this is the most important function in this script, it select the highest prediction from the list of trained dataset and give the results, you'll need to modify this function to adapt to your application, the detect function looks like this:
```
def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    result = model.predict(lm_list)
    percentage_result = result * 100
    print(f"Model prediction result: {percentage_result}")
    if result[0][0] > 0.5:
        label = "not grasped"
    elif result[0][1] > 0.5:
        label = "grasping"
    elif result[0][2] > 0.5:
        label = "carrying"
    elif result[0][3] > 0.5:
        label = "cupping"
    if label in ["grasping", "carrying", "cupping"]:
        label = "grasped"
    return str(label)
```
The variable `result` is a numpy array store the predictions of your model, it's a list in list, for example `[[5.8956690e-02 2.3932913e-05 2.0575800e+00 9.7883438e+01]]`, so the way I did it in my code, is I indexing through the results, and pick the one that is higher than 50% (higher than the rest) as the result, that's why I have a bunch of if functions, you can simply just choose the max value index in the list, but I did this for customization.

The thing you need to change in this function is the label again, go back to step 2, in the categorizing datasets, remember when I told you to pay attention to the `y.attend()` line? Those value that we attend in y represent our label in this function, so you'll need to see what label comes with which y value, and then change the label in this detect function accordingly.  

For example with my sample code, I have:
- neutral: 0
- grasping: 1
- carrying: 2
- cupping: 3

So I fill it according to the `result[0][i]` value, with i is your `y.attend()` variable

Once you go through all of the modifications, you're ready to go, execute the script by:
```
python3 pose_lstm_realtime.py
```
or
```
python3 hand_lstm_realtime.py
```
Optional: You can customize the label, bounding boxes, etc ... with matplotlib to make the detection looks cooler, more sci-fi.  

Troubleshooting: If you have a Sequential-related error, it's very likely that your trained dataset shape and the realtime dataset have different size, for example, you could be training a hands datasets with both hands but when you're doing detection there is only 1 hand, or vice versa. Make sure the size of the dataset and the realtime data matched.

### Demo video:

1. LSTM Pose (Violent Detection): https://youtu.be/Rnu7qdCSr9Q
2. LSTM Hand (Grasp Object): https://youtu.be/ri4uk30wS0A