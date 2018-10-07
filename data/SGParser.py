import json
import pandas as pd

class SGParser:
	def __init__(self,scene_graph):
		print ("inits")
		self.sgraph = scene_graph

		with open(self.sgraph) as sgraph:
			self.sg = json.load(sgraph) #pd.read_json(sgraph)


	def show_sg(self):
		ids =[]
		c = 0
		for rln in self.sg[0]['relationships'][0:5]:
			if rln['subject_id'] not in ids:
				ids.append(rln['subject_id'])
			if rln['object_id'] not in ids:
				ids.append(rln['object_id'])
			print(ids)
			c = c + 1
		return ids

	def show_ob(self,ids):

		for rln in self.sg[0]['objects']:
			ido = rln['object_id']
			if ido in ids:
				print(ido)

if __name__ == '__main__':

	s = SGParser('test_sg.json')

	ids = s.show_sg()
	print "ids : ",len(ids)

