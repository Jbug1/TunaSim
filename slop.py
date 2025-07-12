

import matplotlib.pyplot as plt
import numpy as np
pos = 1
neg = -1
pos_prob = 0.75
neg_prob = 0.25

neg_term = -neg_prob * neg
pos_term = pos_prob * pos
print(neg_term / (pos_term + neg_term), pos_term / (pos_term + neg_term))


a_ = list(np.random.choice([neg, pos], 1000000, p = [neg_prob, pos_prob]))
beta = 0.5

running = (pos + neg) / 2
running_scale = (pos - neg) / 2
step = 1
steps_ = list()
running_direction_ = 0.5
scales_ = list()
directions_ = list()
runnings_ = list()
estimates = list()
pos_ratios_ = list()
scaled_runnings_ = list()
scale_beta = 0.5
overs_ = list()
unders_ = list()
holdover = (pos - neg) / 2
running_scale = holdover

for i in a_:

    running_direction_ = running_direction_ * beta + int(i >= 0) * (1 - beta) 
    running = running * beta + (i) * (1 - beta)

    if i >0:
        overs_.append(running_scale - 4)

    else:
        unders_.append(running_scale - 4)

    if running_direction_ > 0:
        pos_ratio = (running  + (1 - running_direction_)) / running_direction_
    else:
        pos_ratio = 0

    runnings_.append(running)
    directions_.append(running_direction_)
    scales_.append(running_scale)

    if i > 0:
        scaled_runnings_.append(running / (running_scale))
    else:
        scaled_runnings_.append(running / (running_scale))

    pos_ratios_.append(pos_ratio)

    running_scale = running_scale * scale_beta + holdover * (1 -  scale_beta)
    holdover = abs(i)

    # estimated_pos = pos_ratio / (pos_ratio + 1) 
    # estimates.append(estimated_pos)
    # steps_.append(pos_ratio * running_direction_)
    # scales_.append(running_scale)
    
runnings_ = np.array(runnings_)
directions_ = np.array(directions_)
scales_ = np.array(scales_)
scaled_runnings_ = np.array(scaled_runnings_)
pos_ratios_ = np.array(pos_ratios_)


#plt.plot(range(len(steps_)), steps_, c = 'b')
# neg_term = -neg_prob * neg
# pos_term = pos_prob * pos
# a = list(np.random.choice([-1,1], 1000000, p = [neg_term / (pos_term + neg_term), pos_term / (pos_term + neg_term)]))
# print(neg_term / (pos_term + neg_term), pos_term / (pos_term + neg_term))


running = 0

running_std = 1
step = 1
steps = list()
runnings = list()
scales = list()
running_direction = 0.5
# for i in a:


#     running_direction = running_direction * beta + int(i >= 0) * (1 - beta) 
#     running = running * beta + (i) * (1 - beta)
#     runnings.append(running)
#     if running_direction > 0:
#         pos_ratio = (running + (1 - running_direction)) / running_direction
#     else:
#         pos_ratio = 0
#     pos_ratios.append(pos_ratio)
#     estimated_pos = pos_ratio / (pos_ratio + 1) 
#     estimates.append(estimated_pos)
#     running_scale = running_scale * beta + abs(i) * (1 - beta)
#     steps.append(pos_ratio * running_direction)
#     scales.append(running_scale)
    
