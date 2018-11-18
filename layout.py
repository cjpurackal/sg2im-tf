import tensorflow as tf 
tf.enable_eager_execution()




def _boxes_to_grid(boxes, H, W):
  """
  Input:
  - boxes: FloatTensor of shape (O, 4) giving boxes in the [x0, y0, x1, y1]
    format in the [0, 1] coordinate space
  - H, W: Scalars giving size of output
  Returns:
  - grid: FloatTensor of shape (O, H, W, 2) suitable for passing to grid_sample
  """

  O = boxes.shape[0]

  boxes = tf.reshape(boxes,[O,4,1,1])

  x0, y0 = boxes[:, 0], boxes[:, 1]
  x1, y1 = boxes[:, 2], boxes[:, 3]

  ww = x1-x0
  hh = y1-y0

  X = tf.linspace(0, 1, num=W)
  X = tf.reshape(X,[1, 1, w])
  #todo : torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes) --> to boxes part

  Y = tf.linspace(0, 1, num=H)
  Y = tf.reshape(Y,[1, H, 1])


  X = (X - x0)/ww  #(O,1,W)
  Y = (Y - y0)/hh  #(O,H,1)


  X = tf.tile(X,[1,H,1])
  Y = tf.tile(Y,[1,1,W])
  grid = tf.stack([X,Y], axis=3)

  grid =  (grid * 2) - 1


  return grid

if __name__ == '__main__':

 	boxes = tf.ones([10,4])
 	H = W = 64 
 	print(_boxes_to_grid(boxes, H, W))