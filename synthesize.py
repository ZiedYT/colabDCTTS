# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function
import time
import os

from hyperparams import Hyperparams as hp
import hyperparams

import numpy as np

import tensorflow as tf;
from train import Graph
from utils import *
from data_load import load_data ,getNormalText
from scipy.io.wavfile import write
from tqdm import tqdm

def restore(sess,logPath):
	startime = time.time()
	print("LODING SAVED STATE************")
	var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
	saver1 = tf.train.Saver(var_list=var_list)
	saver1.restore(sess, tf.train.latest_checkpoint(logPath + "-1"))
	print("Text2Mel Restored!")

	var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
			   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
	saver2 = tf.train.Saver(var_list=var_list)
	saver2.restore(sess, tf.train.latest_checkpoint(logPath + "-2"))
	print("SSRN Restored!")
	print("restored ",time.time()-startime)
	#saver = tf.train.Saver()
	#saver.save(sess, 'my_test_model',global_step=10000000)
	return sess

def synthesize():
	# Load data
	## use getNormalText(text) to get the input text instead of harvard.txt

	# Load graph
	g = Graph(mode="synthesize"); print("Graph loaded")
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	#sess = tf.Session(config=config)
	##maybe save multiple sess??? L are the lines from the txt
	with tf.Session(config=config) as sess:  ##changes hp.logdir with the dir name for each tf.session ## ex: noelhp = hyperparams.HyperparamsCustom("noel") then use noelhp.logdir
		sess.run(tf.global_variables_initializer())

		sess=restore(sess,hp.logdir)

		while (True):

			val = input("Enter your text: ")
			L = getNormalText(val)
			Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
			prev_max_attentions = np.zeros((len(L),), np.int32)
			for j in tqdm(range(hp.max_T)):
				_gs, _Y, _max_attentions, _alignments = \
					sess.run([g.global_step, g.Y, g.max_attentions, g.alignments],
							 {g.L: L,
							  g.mels: Y,
							  g.prev_max_attentions: prev_max_attentions})
				Y[:, j, :] = _Y[:, j, :]
				prev_max_attentions = _max_attentions[:, j]

			# Get magnitude
			Z = sess.run(g.Z, {g.Y: Y})

			# Generate wav files
			if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
			for i, mag in enumerate(Z):
				print("Working on file", i+1)
				wav = spectrogram2wav(mag)
				write(hp.sampledir + "/{}.wav".format(i+1), hp.sr, wav)

if __name__ == '__main__':
	synthesize()
	print("Done")


