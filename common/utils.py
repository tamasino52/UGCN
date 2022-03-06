import torch
import numpy as np
import hashlib
from torch.autograd import Variable
import os
import matplotlib.pyplot as plt

def show3DposePair(realt3d, faket3d, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=True,
                   gt=True, pred=False):  # blue, orange
  """
  Visualize a 3d skeleton pair

  Args
  channels: 96x1 vector. The pose to plot.
  ax: matplotlib 3d axis to draw on
  lcolor: color for left part of the body
  rcolor: color for right part of the body
  add_labels: whether to add coordinate labels
  Returns
  Nothing. Draws on ax.
  """
  #   assert channels.size == len(data_utils.H36M_NAMES)*3, "channels should have 96 entries, it has %d instead" % channels.size
  realt3d = np.reshape(realt3d, (17, -1))
  faket3d = np.reshape(faket3d, (17, -1))

  I = np.array([0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15])  # start points
  J = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])  # end points
  LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

  for idx, vals in enumerate([realt3d, faket3d]):
    # Make connection matrix
    for i in np.arange(len(I)):
      x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
      if idx == 0:
        ax.plot(x, z, -y, lw=2, c='k')
      #        ax.plot(x,y, z,  lw=2, c='k')

      elif idx == 1:
        ax.plot(x, z, -y, lw=2, c='r')
      #        ax.plot(x,y, z,  lw=2, c='r')

      else:
        #        ax.plot(x,z, -y,  lw=2, c=lcolor if LR[i] else rcolor)
        ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

  RADIUS = 1  # space around the subject
  xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
  ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
  ax.set_ylim3d([-RADIUS+zroot, RADIUS+zroot])
  ax.set_zlim3d([-RADIUS-yroot, RADIUS-yroot])

  if add_labels:
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("-y")

  # Get rid of the ticks and tick labels
  #  ax.set_xticks([])
  #  ax.set_yticks([])
  #  ax.set_zticks([])
  #
  #  ax.get_xaxis().set_ticklabels([])
  #  ax.get_yaxis().set_ticklabels([])
  #  ax.set_zticklabels([])
  #     ax.set_aspect('equal')

  # Get rid of the panes (actually, make them white)
  white = (1.0, 1.0, 1.0, 0.0)
  ax.w_xaxis.set_pane_color(white)
  ax.w_yaxis.set_pane_color(white)
  # Keep z pane

  # Get rid of the lines in 3d
  ax.w_xaxis.line.set_color(white)
  ax.w_yaxis.line.set_color(white)
  ax.w_zaxis.line.set_color(white)


def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2 ** 32 - 1) * (max_value - min_value)) + min_value


def mpjpe_cal(predicted, target):
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1))


def test_calculation(predicted, target, action, error_sum, data_type, subject):
    error_sum = mpjpe_by_action(predicted, target, action, error_sum)

    return error_sum


def mpjpe_by_action(predicted, target, action, action_error_sum):
    assert predicted.shape == target.shape
    num = predicted.size(0)
    dist = torch.mean(torch.norm(predicted - target, dim=len(target.shape) - 1), dim=len(target.shape) - 2)

    if len(set(list(action))) == 1:
        end_index = action[0].find(' ')
        if end_index != -1:
            action_name = action[0][:end_index]
        else:
            action_name = action[0]

        action_error_sum[action_name].update(torch.mean(dist).item()*num, num)
    else:
        for i in range(num):
            end_index = action[i].find(' ')
            if end_index != -1:
                action_name = action[i][:end_index]
            else:
                action_name = action[i]

            action_error_sum[action_name].update(dist[i].item(), 1)
            
    return action_error_sum


def define_actions( action ):

  actions = ["Directions","Discussion","Eating","Greeting",
           "Phoning","Photo","Posing","Purchases",
           "Sitting","SittingDown","Smoking","Waiting",
           "WalkDog","Walking","WalkTogether"]

  if action == "All" or action == "all" or action == '*':
    return actions

  if not action in actions:
    raise( ValueError, "Unrecognized action: %s" % action )

  return [action]


def define_error_list(actions):
    error_sum = {}
    error_sum.update({actions[i]: AccumLoss() for i in range(len(actions))})
    return error_sum


class AccumLoss(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        

def get_varialbe(split, target):
    num = len(target)
    var = []
    if split == 'train':
        for i in range(num):
            temp = Variable(target[i], requires_grad=False).contiguous().type(torch.cuda.FloatTensor)
            var.append(temp)
    else:
        for i in range(num):
            temp = Variable(target[i]).contiguous().cuda().type(torch.cuda.FloatTensor)
            var.append(temp)

    return var


def print_error(data_type, action_error_sum, is_train):
    mean_error = print_error_action(action_error_sum, is_train)

    return mean_error


def print_error_action(action_error_sum, is_train):
    mean_error_each = 0.0
    mean_error_all = AccumLoss()

    if is_train == 0:
        print("{0:=^12} {1:=^10}".format("Action", "MPJPE"))

    for action, value in action_error_sum.items():
        if is_train == 0:
            print("{0:<12} ".format(action), end="")
            
        mean_error_each = action_error_sum[action].avg * 1000.0
        mean_error_all.update(mean_error_each, 1)

        if is_train == 0:
            print("{0:>7.2f}".format(mean_error_each))

    if is_train == 0:
        print("{0:<12} {1:>7.2f}".format("Average", mean_error_all.avg))
    
    return mean_error_all.avg


def save_model(previous_name, save_dir, epoch, data_threshold, model):
    if os.path.exists(previous_name):
        os.remove(previous_name)

    torch.save(model.state_dict(),
               '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100))
    previous_name = '%s/model_%d_%d.pth' % (save_dir, epoch, data_threshold * 100)
    return previous_name
    

def motion_cal(predicted, target, intervals=[8, 12, 16, 24], operator=torch.cross):
    assert predicted.shape == target.shape
    loss = 0
    for itv in intervals:
        pred_encode = operator(predicted[:, :-itv], predicted[:, itv:], dim=3)
        target_encode = operator(target[:, :-itv], target[:, itv:], dim=3)
        loss += torch.mean(torch.abs(pred_encode - target_encode)) / len(intervals)
    return loss






