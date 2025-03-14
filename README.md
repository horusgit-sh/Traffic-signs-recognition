That is Neural Network Model to classify road signs based on an image of those signs. 

HOW TO USE:

user@kali:~$ git clone https://github.com/horusgit-sh/Traffic-signs-recognition.git

user@kali:~$ pip install -r requirements.txt

user@kali:~$ python3 traffic.py data_directory

What did I do:

In first version was only one neural layer with Batch Normalization. Accuracy was ~5%, loss = 3.5.
Not good at all.

My second try was more successful. I add 2 more neural layers with Batch Normalization, Global Average Pooling and less Dropout. 
Accuracy was ~97%!!, loss = 0.08.

Finally, I decided to add one more neural layer, and Accuracy became 99,2% with 0.04 loss.


