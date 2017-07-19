#-*- coding:utf-8 -*-
"""
封装训练好的model
"""

import tensorflow as tf

class Model(object):

	def __init__(self, model_dir, tags, signature_key, input_keys, output_keys):
		self.sess = tf.Session()
		self.meta_graph_def = tf.saved_model.loader.load(self.sess, tags, model_dir)
		self.signature = self.meta_graph_def.signature_def

		# input用map,后续要将实际输入的数据与tensor_name匹配构造feed_dict 
		self.key_tensor_name_map = {}
		for key in input_keys:
			tensor_name = self.signature[signature_key].inputs[key].name
			self.key_tensor_name_map[key] = tensor_name

		self.output_tensor_names = []
		for key in output_keys:
			tensor_name = self.signature[signature_key].outputs[key].name
			self.output_tensor_names.append(tensor_name)


	def predict(self, inputs):
		"""
		- inputs : a map like {input_key1: data ,  input_key2: data, ... input_keyn: data}
		"""
		if not isinstance(inputs, dict):
			raise Exception('inputs must be a dict, like {input_key1: data ,  input_key2: data, ... input_keyn: data}')

		feed_dict = {}
		for key in inputs:
			feed_dict[self.key_tensor_name_map[key]] = inputs[key]

		results = self.sess.run(self.output_tensor_names, feed_dict=feed_dict)
		return results


if __name__ == '__main__':
	model = Model('test_saved_model', ['test_saved_model'], 'test_signature', ['input_x', 'keep_prob'], ['output'])
	inputs = {'input_x': [[1, 208, 208]], 'keep_prob': 1.0}
	results = model.predict(inputs)
	print results