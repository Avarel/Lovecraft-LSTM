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
# Instructions
## Setting Up the Code

### Local Machine
* Clone the repository into your local machine.

### Google Colaboratory
* Upload the training and validation text files.
    * Compress the contents of the project into a `.zip` and extract using the following command on the kernel.
        * `!unzip -o "./Archive.zip" -d "./"`
* Set up the journal to use the correct Tensorflow version and beg the Google lords to allow you to use their **Nvidia Tesla P100** GPUs.
    * Add the following into the kernel:
         ```python
            try:
                %tensorflow_version 2.x
            except Exception:
                pass
            !pip install tensorflow-gpu```
* Copy the code from `./main.py` over to a code block.
* Run the code block.

## Running the Code
### Local Machine
* Unless you have a strong GPU for machine learning, good luck.
* You can, however, download a checkpoint and generate text with it. In that case, you can read the dataset, build a model, load the weights, and generate the text with the Google Colaboratory instructions.

### Google Colaboratory
For each of the desired tasks, type the following code blocks into code cells on the notebook and execute.

#### Reading the Dataset
```python
import ml_data
vocab_len, char2int, int2char, tr_dataset, val_dataset = ml_data.get_datasets(batch_size)
```
#### Building the Model
```python
import ml
model = ml.build_model(
    vocab_len,
    embedding_dim,
    rnn_units,
    batch_size
)
```

#### Training the Model
```python
import ml
ml.train_model(model, tr_dataset, val_dataset, checkpoint_dir)
```

#### Resuming from a Checkpoint
```python
import ml
import tensorflow as tf
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
```

#### Generating Text with the Model
```python
import ml
ml.generate_text(model, char2int, int2char, "the deep dark")
```

#### Downloading Checkpoints

* Use the following lines.
    ```python
    from google.colab import files
    files.download('./checkpoints/cp_XXXX/ckpt_XX.data-0000X-of-0000X') 
    ```

# Sample
Here's a sample so far! Nothing interesting though. Need to feed more data into the model.

`
the deep dark whe did not like the way the spell, though whether they were intimated from the southeast. Only when they approached a display sane polar which such continues and filling in this air—bat anyway at the sea. S’ine of the carried city ahead. The witch Tilts were left and extent, the other brown. They were the prismatic seal in foetid alpost at the hills through entity in the great black hands in the Subodden sun talking with a paising rose above a huge gradual levels he did not place it, I would kind o’ believe that was not to exert 5 amidst its fiture and almost brainless thick and managing the daemoniac framework of reporters who keep the ruins on the water and saw that they were still thinking about a hill and inious that might surge out the glass from the alley-turned handlight ahead as it shortly previnced, details. Some of these times life the lurking men’s seeking ahead of various things. When it could not be put into a form after I could come down the small skull where the thing 
`