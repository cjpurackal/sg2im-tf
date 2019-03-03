import tensorflow as tf 
import utils.bilinear_sampler as bs


def make_from_boxes(boxes, vi, H, W):
  grid = _boxes_to_grid(boxes, H, W)
  O = vi.shape[0]
  D = vi.shape[1]
  vi = tf.tile(tf.reshape(vi, [O, 1, 1, D]), [1, 8, 8, 1])
  print (vi.shape)
  print (grid.shape)
  # need to reimplement bilinear_sampler
  sampled = bs.bilinear_sampler(vi, grid[:,:,:,0], grid[:,:,:,1])
  print (sampled.shape)

def _boxes_to_grid(boxes, H, W):
  O = boxes.shape[0]

  boxes = tf.reshape(boxes,[O,4,1,1])

  x0, y0 = boxes[:, 0], boxes[:, 1]
  x1, y1 = boxes[:, 2], boxes[:, 3]

  ww = x1-x0
  hh = y1-y0

  X = tf.linspace(0.0, 1.0, num=W)
  X = tf.reshape(X,[1, 1, W])

  Y = tf.linspace(0.0, 1.0, num=H)
  Y = tf.reshape(Y,[1, H, 1])


  X = (X - x0)/ww  #(O,1,W)
  Y = (Y - y0)/hh  #(O,H,1)


  X = tf.tile(X,[1,H,1])
  Y = tf.tile(Y,[1,1,W])
  grid = tf.stack([X,Y], axis=3)

  grid =  (grid * 2) - 1


  return grid

