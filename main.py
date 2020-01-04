import ml
import ml_data
import os

import tensorflow as tf

# Persistent variables that I have not yet compartmentalize.
batch_size: int = 64
embedding_dim: int = 256
rnn_units: int = 4096
checkpoint_dir = os.path.join(".", "checkpoints", 'cp_x')

# Grab the datasets.
vocab_len, char2int, int2char, tr_dataset, val_dataset = ml.get_datasets(batch_size)

# Build the model.
# model = ml.build_model(
#     vocab_len,
#     embedding_dim,
#     rnn_units,
#     batch_size
# )
# model.summary()

# Train the model.
# ml.train_model(model,
#                tr_dataset, val_dataset, checkpoint_dir,
#                epochs=500, patience=25, save_one=True)

# Reconstruct and feed the checkpoint data into the model.
model = ml.build_model(vocab_len, embedding_dim, rnn_units, 1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

print(ml.generate_text(model, char2int, int2char, "The Call of Cthulhu.", 50))