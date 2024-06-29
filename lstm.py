# This module prepares midi file data and feeds it to the neural
# network for training

import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def train_network():
    # Train a Neural Network to generate music
    notes = get_notes()

    # Get number of unique pitch names
    n_vocab = len(set(notes))

    # Prepare sequences for the network
    network_input, network_output = prepare_sequences(notes, n_vocab)

    # Create the LSTM network model
    model = create_network(network_input, n_vocab)

    # Train the model
    train(model, network_input, network_output)

def get_notes():
    # Get all the notes and chords from the MIDI files in the ./midi_songs directory
    notes = []

    # Iterate through all MIDI files in the directory
    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)
        print("Parsing %s" % file)

        # Check if MIDI file has instrument parts
        try:
            # Partition by instrument
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            # If file has notes in a flat structure
            notes_to_parse = midi.flat.notes

        # Store pitch and chord information
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    # Save notes data to a file
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    # Prepare the sequences used by the Neural Network
    sequence_length = 100  # Length of input sequences

    # Get all unique pitch names and sort them
    pitchnames = sorted(set(item for item in notes))

    # Create a dictionary mapping pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # Create input sequences and corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    # Number of input patterns
    n_patterns = len(network_input)

    # Reshape input into a format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalize input
    network_input = network_input / float(n_vocab)

    # One-hot encode the output
    network_output = np_utils.to_categorical(network_output)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    # Create the structure of the neural network
    model = Sequential()

    # Add LSTM layers with dropout and batch normalization
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))

    # Add dense layers with activation and batch normalization
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))

    # Output layer with softmax activation for probability distribution
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))

    # Compile model with categorical cross-entropy loss and RMSprop optimizer
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output):
    # Train the neural network
    # Define checkpoint to save the best model during training
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    # Fit the model on training data
    model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)

if __name__ == '__main__':
    train_network()
