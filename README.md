# Lovecraft-LSTM

A machine learning (Tensorflow) program that generates Lovecraftian horrors for your horror obsession with the old gods, who will remake this world in their image when they awake from their slumber. Based on [this tutorial](https://towardsdatascience.com/generating-text-with-tensorflow-2-0-6a65c7bdc568) on LSTMs.

# Tensorflow Model
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (64, None, 256)           23040     
_________________________________________________________________
dropout_3 (Dropout)          (64, None, 256)           0         
_________________________________________________________________
lstm_2 (LSTM)                (64, None, 1024)          5246976   
_________________________________________________________________
dropout_4 (Dropout)          (64, None, 1024)          0         
_________________________________________________________________
lstm_3 (LSTM)                (64, None, 1024)          8392704   
_________________________________________________________________
dropout_5 (Dropout)          (64, None, 1024)          0         
_________________________________________________________________
dense_1 (Dense)              (64, None, 90)            92250     
=================================================================
Total params: 13,683,234
Trainable params: 13,683,234
Non-trainable params: 0
_________________________________________________________________
```

# Running the Code
## Local Machine
* One day, I'll get a GPU strong enough that I can actually do this, but not today.

## Google Colaboratory
* Upload the training and validation text files.
    * Compress the contents of the folders into a `.zip` and extract using the following command on the kernel.
        * `!unzip -o "./data.zip" -d "./"`
* Set up the journal to use the correct Tensorflow version and beg the Google lords to allow you to use their **Nvidia Tesla P100** GPUs.
    * Add the following into the kernel:
         ```python
            try:
                %tensorflow_version 2.x
            except Exception:
                pass
            !pip install tensorflow-gpu```
* Copy the code from `./main.py` over to a code block.
* Run the code block first, this is very important so the journal is set up with the right code.
* Run `generate()`.

# Sample
Here's a sample so far! Nothing interesting though. Need to feed more data into the model.
```txt
em hen the bring of Dmick the Tompore there, and tryemitable where and with the own went Mulhage, but tridge under my Surhas had seen not the woods and unvising hingleas feor the threo. So thlow! As the space could have abaeanted for the cover roundath a gepen suddenly after all tremended and reflective space to the Old Orden what its fasust fize when he perided Willges from Duals Atuma he had been pereptted. Srepent case the blushic when the eusher sud along at that millings. Ind adving, dreams and cembles and flightened by what the lind and exrivible in which had terrible the branwing was from Ashemian Jambar, refementiting, and the other, most that we Bake out a sizen age. The though prespented arr one mullis, an’ very mysenices all in the fa
```