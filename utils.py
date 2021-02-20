import numpy as np
import torch
import shutil
import cv2

def decode_segmap(image, nc=3):
  """To color code the semantic map (output)"""
  
  label_colors = np.array([ (0, 0, 0),   # 0 = background
                            (128, 0, 0), # 1 = wheel
                            (0, 128, 0)  # 2 = frame
                          ])

  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
  
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
    
  rgb = np.stack([r, g, b], axis=3)
  return rgb


def to_device(data, device):
  """Move tensor(s) to chosen device"""
  
  if isinstance(data, (list,tuple)):
    return [to_device(x, device) for x in data]
  
  return data.to(device, non_blocking=True)


def iou_numpy(outputs: np.array, labels: np.array):
  """To calculate IOU for each label"""
    
  # To avoid division by zero
  SMOOTH = 1e-6   

  intersection = (outputs & labels).sum((0, 1))
  union = (outputs | labels).sum((0, 1))
  iou = float(intersection + SMOOTH) / (union + SMOOTH)
    
  return iou


def IOU(target,pred,num_classes=3):
  """To calculate mean IoU"""

  a = np.zeros((num_classes,), dtype=float)

  for i in range(num_classes):
    target_mask = (target == i)
    pred_mask = (pred ==i)
    a[i] = iou_numpy(target_mask, pred_mask)
    
  return a


def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    # save checkpoint data to the path given, checkpoint_path
    torch.save(state, f_path)
    # if it is a best model, min validation loss
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)


def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    # load check point
    checkpoint = torch.load(checkpoint_fpath)
    # initialize state_dict from checkpoint to model
    model.load_state_dict(checkpoint['state_dict'])
    # initialize optimizer from checkpoint to optimizer
    optimizer.load_state_dict(checkpoint['optimizer'])
    # initialize valid_loss_min from checkpoint to valid_loss_min
    valid_loss_min = checkpoint['valid_loss_min']
    # return model, optimizer, epoch value, min validation loss 
    return model, optimizer, checkpoint['epoch'], valid_loss_min


def count_blobs_w(img):
    """To count the number of wheels in given output"""

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opening = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=5)
    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    blobs = 0

    for c in cnts:
        area = cv2.contourArea(c)
        if area > 5000 and area < 31000:
            blobs += 1

    return blobs


def count_blobs_f(img):
    """To count the number of frames in given output"""

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    opening = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=7)

    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    blobs = 0

    for c in cnts:
        area = cv2.contourArea(c)
        if area > 5000 and area < 46000:
            blobs += 1

    return blobs


def counter(img):
    """To count number of components in the output image"""

    b, g,  _ = cv2.split(img)

    wheels = b
    idx = wheels > 0
    wheels[idx] = 255

    num_wheels = count_blobs_w(wheels)

    frames = g
    idx = frames > 0
    frames[idx] = 255

    num_frames = count_blobs_f(frames)

    return [num_wheels, num_frames]


def print_rule(X_test,clf):
  """To print the logic flow (interpretation) acquired from the decision tree"""

  node_indicator = clf.decision_path(X_test)
  leaf_id = clf.apply(X_test)
  feature = clf.tree_.feature
  threshold = clf.tree_.threshold
  feature_names = ['Num_Wheels','Num_Frames']
  class_names = ['neither','unicycle','bicycle']

  sample_id = 0

  # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
  node_index = node_indicator.indices[node_indicator.indptr[sample_id]:
                                        node_indicator.indptr[sample_id + 1]]

  print('Rules used to predict given sample:\n')
  for node_id in node_index:
      # continue to the next node if it is a leaf node
      if leaf_id[sample_id] == node_id:
          continue

      # check if value of the split feature for sample 0 is below threshold
      if (X_test[0][feature[node_id]] <= threshold[node_id]):
          threshold_sign = "<="
      else:
          threshold_sign = ">"

      print("Decision Node {node} : ({feature} = {value}) "
            "{inequality} {threshold})".format(
                node=node_id,
                feature=feature_names[feature[node_id]],
                value=X_test[0][feature[node_id]],
                inequality=threshold_sign,
                threshold=threshold[node_id]))
  print("Predicted Class : {}\n".format(class_names[int(clf.predict(X_test))]))