import os
import wave
import pyaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


class GradientSonification(Callback):
    """ Convert the norm of the gradients into audio """

    def __init__(self,
                 path,
                 model,
                 fs=44100,
                 duration=0.01,
                 freq=200.0):

        self.path = path if path.endswith('.wav') else path + '.wav'
        self.model = model
        self.trainable_layers = [layer.name for layer in self.model.layers if layer.trainable_weights != []]
        self.metrics = [self.get_metrics(layer) for layer in self.trainable_layers]
        self.fs = fs
        self.duration = duration
        self.freq = freq
        self.frames = []


    def get_metrics(self, layer):
        ''' Create a custom metric which outputs the gradient norm for a given layer '''
        def func(y_true, y_pred):
            grad = self.model.optimizer.get_gradients(y_pred, self.model.get_layer(layer).trainable_weights[0])[0]
            norm = K.sqrt(K.sum(K.square(grad)))
            return norm

        metric = func
        metric.__name__ = 'gradient_norm_' + layer
        return metric


    def on_train_begin(self, logs={}):
        self.p, self.stream = self.open_stream()


    def on_train_end(self, logs={}):
        ''' Save the frames to a wav file '''
        wf = wave.open(self.path, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(self.p.get_sample_size(pyaudio.paFloat32))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(self.frames))
        wf.close()
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()
        self.plot_audio()

    def on_batch_end(self, batch, logs={}):
        '''
        Since the gradient norms are metrics they will be stored in the logs
        dict with keys equivalent to the function __name__ attribute
        '''
        for layer in self.trainable_layers:
            tone = self.freq + ((logs.get('gradient_norm_' + layer)) * 100.0)
            tone = tone.astype(np.float32)
            samples = self.generate_tone(tone)
            self.frames.append(samples)

        # Insert silence sample between batches
        silence = np.zeros(samples.shape[0] * 2, dtype=np.float32)
        self.frames.append(silence)


    def open_stream(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paFloat32,
                        channels=1,
                        rate=self.fs,
                        output=True)
        return p, stream


    def generate_tone(self, tone):
        npsin = np.sin(2 * np.pi * np.arange(self.fs*self.duration) * tone / self.fs)
        samples = npsin.astype(np.float32)
        return 0.1 * samples


    def plot_audio(self):
        ''' Plot waveplot and spectrogram of recording '''
        x, sr = librosa.load(self.path)

        # Wave plot
        fname = os.splitext(self.path)[0] + '_waveplot.png'
        plt.figure(figsize=(14, 5))
        librosa.display.waveplot(x, sr=sr)
        plt.savefig(fname)

        # Spectrogram
        fname = os.splitext(self.path)[0] + '_spectrogram.png'
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))
        plt.figure(figsize=(14, 5))
        librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
        plt.colorbar()
        plt.savefig(fname)