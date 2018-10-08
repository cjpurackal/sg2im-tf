import json

def get_edges(s_graph):
	with open(s_graph) as sgraph:
		sg = json.load(sgraph) 	 
	edges = []
	for s in sg:
		for rln in s['relationships']:
			edges.append((rln['object_id'],rln['relationship_id'],rln['subject_id']))
	return edges


def get_objects_unique(s_graph):
	with open(s_graph) as sgraph:
		sg = json.load(sgraph) 	
	ids = []	
	for s in sg:
		for rln in s['relationships']:
			if rln['subject_id'] not in ids:
				ids.append(rln['subject_id'])
			if rln['object_id'] not in ids:
				ids.append(rln['object_id'])

	return ids


if __name__ == '__main__':


	e = get_edges('test_sg.json')
	for i in e:
		print i

	e = get_objects_unique('test_sg.json')
	for i in e:
		print i