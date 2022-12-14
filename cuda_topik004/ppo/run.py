# Reference: https://github.com/UesugiErii/tf2-mpi-ppo
# Modified to run using GPU

import numpy as np
from datetime import datetime
from atari_wrappers import *
import gym
import time

size = 1

# placeholder
send_state_buf = None  # use to send state
recv_state_buf = None  # use to recv state
r = None  # reward
done = None  # done
info = None  # info
a = None  # action

# parameter
work_dir = os.path.dirname(os.path.abspath(__file__))
logdir = work_dir + "/logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
weight_dir = work_dir + "/logs/weight/" + datetime.now().strftime("%Y%m%d-%H%M%S") + '/'
learning_rate = 0.00025
is_annealed = False  # Decide whether the learning rate decreases linearly
total_step = 1 * 10 ** 7
# total_step = 6 * 10 ** 5
clip_epsilon = 0.1
# epochs = 128  # horizon
epochs = 512  # horizon
# use_RNN = True
use_RNN = False

beta = 0.01
VFcoeff = 1
IMG_W = 84  # image width
IMG_H = 84  # image height
hidden_unit_num = 128
if use_RNN:
    k = 1
else:
    k = 4
env_name = 'PongDeterministic-v4'  # env name
# env_name = 'Frostbite-v0'  # env name

# Dynamically get the number of actions
env = gym.make(env_name)
env = WarpFrame(env, width=IMG_W, height=IMG_H, grayscale=True)
env = FrameStack(env, k=k)  # return (IMG_H , IMG_W ,k)
a_num = env.action_space.n

# batch_size = epochs * size // 4
batch_size = 128
gamma = 0.99  # discount reward

# print(f"epochs: {epochs} | size: {size} | batch_size: {batch_size}")

# Pretending as master
rank = 0
# brain
if rank == 0:
    #######################
    # calc real v and adv #
    #######################

    import numba


    @numba.jit(nopython=True)
    def calc_real_v_and_adv_GAE(v, r, done):
        length = r.shape[0]
        num = r.shape[1]

        adv = np.zeros((length + 1, num), dtype=np.float32)

        for t in range(length - 1, -1, -1):
            delta = r[t, :] + v[t + 1, :] * gamma * (1 - done[t, :]) - v[t, :]
            adv[t, :] = delta + gamma * 0.95 * adv[t + 1, :] * (1 - done[t, :])  # 0.95 is lambda

        adv = adv[:-1, :]

        realv = adv + v[:-1, :]

        return realv, adv


    @numba.jit(nopython=True)
    def calc_real_v_and_adv(v, r, done):
        length = r.shape[0]
        num = r.shape[1]

        realv = np.zeros((length + 1, num), dtype=np.float32)
        adv = np.zeros((length, num), dtype=np.float32)

        realv[-1, :] = v[-1, :] * (1 - done[-1, :])

        for t in range(length - 1, -1, -1):
            realv[t, :] = realv[t + 1, :] * gamma * (1 - done[t, :]) + r[t, :]
            adv[t, :] = realv[t, :] - v[t, :]

        return realv[:-1, :], adv  # end_v dont need


    ###################
    # TensorFlow Part #
    ###################

    import tensorflow as tf
    import os

    # os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    # my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
    # tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
    from tensorflow.python.keras import Model
    from tensorflow.python.keras.layers import Dense, Conv2D, Flatten, LSTMCell
    import tensorflow.keras.optimizers as optim

    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()


    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, init_lr):
            super(CustomSchedule, self).__init__()
            self.lr = init_lr
            self.max_learning_times = total_step * 3 // epochs // (size)

        def __call__(self, step):
            # step start from 0
            # every time call , step automatic += 1
            self.last_lr = self.lr * ((self.max_learning_times - step) / self.max_learning_times)
            return self.last_lr

        def get_config(self):
            return self.last_lr


    if is_annealed:
        optimizer = optim.Adam(learning_rate=CustomSchedule(learning_rate))  # linearly annealed
    else:
        optimizer = optim.Adam(learning_rate=learning_rate)  # no annealed


    class CNNModel(Model):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.c1 = Conv2D(32, kernel_size=(8, 8), strides=(4, 4),
                             activation='relu')
            self.c2 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu')
            self.c3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')
            self.flatten = Flatten()
            self.d1 = Dense(512, activation="relu")
            self.d2 = Dense(1)  # C
            self.d3 = Dense(a_num, activation='softmax')  # A
            self.call(np.random.random((epochs, IMG_H, IMG_W, k)).astype(np.float32))

        @tf.function
        def call(self, inputs):
            x = inputs / 255.0
            x = self.c1(x)
            x = self.c2(x)
            x = self.c3(x)
            x = self.flatten(x)
            x = self.d1(x)
            ap = self.d3(x)
            v = self.d2(x)
            return ap, v

        @tf.function
        def loss(self, state, action_index, adv, real_v, old_ap):
            res = self.call(state)
            error = res[1][:, 0] - real_v
            L = tf.reduce_sum(tf.square(error))

            adv = tf.dtypes.cast(tf.stop_gradient(adv), tf.float32)
            batch_size = state.shape[0]
            all_act_prob = res[0]
            selected_prob = tf.reduce_sum(action_index * all_act_prob, axis=1)
            old_prob = tf.reduce_sum(action_index * old_ap, axis=1)

            r = selected_prob / (old_prob + 1e-6)

            H = -tf.reduce_sum(all_act_prob * tf.math.log(all_act_prob + 1e-6))

            Lclip = tf.reduce_sum(
                tf.minimum(
                    tf.multiply(r, adv),
                    tf.multiply(
                        tf.clip_by_value(
                            r,
                            1 - clip_epsilon,
                            1 + clip_epsilon
                        ),
                        adv
                    )
                )
            )

            return -(Lclip - VFcoeff * L + beta * H) / batch_size

        @tf.function
        def train(self, batch_state, batch_a, batch_adv, batch_real_v, batch_old_ap):
            with tf.GradientTape() as tape:
                loss_value = self.loss(batch_state, batch_a, batch_adv, batch_real_v, batch_old_ap)

            grads = tape.gradient(loss_value, self.trainable_weights)
            grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            return loss_value


    class RNNModel(Model):
        def __init__(self):
            super(RNNModel, self).__init__()
            self.c1 = Conv2D(32, kernel_size=(8, 8), strides=(4, 4),
                             activation='relu')
            self.c2 = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu')
            self.c3 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu')
            self.flatten = Flatten()
            self.d1 = Dense(512, activation="relu")
            self.d2 = Dense(1)  # C
            self.d3 = Dense(a_num, activation='softmax')  # A
            self.lstm_cell = LSTMCell(hidden_unit_num)
            self.call(
                np.random.random((batch_size, IMG_H, IMG_W, k)).astype(np.float32),
                tf.convert_to_tensor(np.zeros((batch_size, hidden_unit_num), dtype=np.float32)),
                tf.convert_to_tensor(np.zeros((batch_size, hidden_unit_num), dtype=np.float32))
            )

        @tf.function
        def call(self, inputs, h, c):
            x = inputs / 255.0
            x = self.c1(x)
            x = self.c2(x)
            x = self.c3(x)
            x = self.flatten(x)
            x = self.d1(x)
            x, hc = self.lstm_cell(inputs=x, states=(h, c))
            a = self.d3(x)
            v = self.d2(x)
            #            h      c
            return a, v, hc[0], hc[1]

        def loss(self, state, action_index, adv, real_v, old_ap, h, c):
            res = self.call(state, h, c)
            error = res[1][:, 0] - real_v
            L = tf.reduce_sum(tf.square(error))

            adv = tf.dtypes.cast(tf.stop_gradient(adv), tf.float32)
            batch_size = state.shape[0]
            all_act_prob = res[0]
            selected_prob = tf.reduce_sum(action_index * all_act_prob, axis=1)
            old_prob = tf.reduce_sum(action_index * old_ap, axis=1)

            r = selected_prob / (old_prob + 1e-6)

            H = -tf.reduce_sum(all_act_prob * tf.math.log(all_act_prob + 1e-6))

            Lclip = tf.reduce_sum(
                tf.minimum(
                    tf.multiply(r, adv),
                    tf.multiply(
                        tf.clip_by_value(
                            r,
                            1 - clip_epsilon,
                            1 + clip_epsilon
                        ),
                        adv
                    )
                )
            )

            return -(Lclip - VFcoeff * L + beta * H) / batch_size

        @tf.function
        def train(self, batch_state, batch_a, batch_adv, batch_real_v, batch_old_ap, batch_h, batch_c):
            with tf.GradientTape() as tape:
                loss_value = self.loss(batch_state, batch_a, batch_adv, batch_real_v, batch_old_ap, batch_h, batch_c)

            grads = tape.gradient(loss_value, self.trainable_weights)
            grads, grad_norm = tf.clip_by_global_norm(grads, 0.5)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            return loss_value

    # Start time counter
    total_time_counter = 0
    total_time_counter -= time.time()

    if use_RNN:
        model = RNNModel()
    else:
        model = CNNModel()

    ########################
    # define some variable #
    ########################

    # if use_RNN:
    #     total_h = np.empty((epochs + 1, size, hidden_unit_num), dtype=np.float32)
    #     total_h[0] = 0
    #     total_c = np.empty((epochs + 1, size, hidden_unit_num), dtype=np.float32)
    #     total_c[0] = 0
    # total_state = np.empty((epochs, size, IMG_H, IMG_W, k), dtype=np.float32)
    if use_RNN:
        total_h = np.empty((epochs//size + 1, size, hidden_unit_num), dtype=np.float32)
        total_h[0] = 0
        total_c = np.empty((epochs//size + 1, size, hidden_unit_num), dtype=np.float32)
        total_c[0] = 0
    total_state = np.empty((epochs//size, size, IMG_H, IMG_W, k), dtype=np.float32)
    # print(f"total_state definition: {total_state.shape}")
    # total_v = np.empty((epochs + 1, size), dtype=np.float32)
    # total_a = np.empty((epochs, size), dtype=np.int32)
    # total_r = np.zeros((epochs, size), dtype=np.float32)
    # total_done = np.zeros((epochs, size), dtype=np.float32)
    # total_old_ap = np.zeros((epochs, size, a_num), dtype=np.float32)  # old action probability
    # recv_state_buf = np.empty((size, IMG_H, IMG_W, k), dtype=np.float32)  # use to recv data
    total_v = np.empty((epochs//size + 1, size), dtype=np.float32)
    total_a = np.empty((epochs//size, size), dtype=np.int32)
    total_r = np.zeros((epochs//size, size), dtype=np.float32)
    total_done = np.zeros((epochs//size, size), dtype=np.float32)
    total_old_ap = np.zeros((epochs//size, size, a_num), dtype=np.float32)  # old action probability
    recv_state_buf = np.empty((size, IMG_H, IMG_W, k), dtype=np.float32)  # use to recv data

    learning_step = 0
    remain_step = total_step // size

    all_reward = np.zeros((size,), dtype=np.float32)  # Used to record the reward of each episode
    one_episode_reward_index = 0  # all env episode index in tensorboard
    count_episode = [0] * size  # count episode index in every env

    ####################
    # brain's env init #
    ####################

    # random init
    np.random.seed(rank)
    env.seed(rank)

    state = np.array(env.reset(), dtype=np.float32)
    send_state_buf = state
    recv_state_buf = np.expand_dims(state, axis=0)

    ###########################
    #      loop               #
    #      ??? <-------- ???      #
    #      1 -> 255 -> 1      #
    ###########################

    # first one
    total_state[0, :, :, :, :] = recv_state_buf
    remain_step -= 1

    if use_RNN:
        ap, v, total_h[1], total_c[1] = model(total_state[0], total_h[0], total_c[0])
    else:
        ap, v = model(recv_state_buf)
    ap = ap.numpy()
    v = v.numpy()
    v.resize((size,))
    total_v[0, :] = v
    total_old_ap[0, :] = ap

    # scattering action
    a = [np.random.choice(range(a_num), p=ap[i]) for i in range(size)]
    total_a[0, :] = a

    # brain's env get first action
    state_, r, done, info = env.step(a)
    # print(f"r alone: {r}")
    if done:
        state_ = np.array(env.reset(), dtype=np.float32)
    state = np.array(state_, dtype=np.float32)
    send_state_buf = state

    # recv other information
    # print(f"r togather: {r}")

    total_r[0, :] = np.array(r, dtype=np.float32)
    total_done[0, :] = np.array(done, dtype=np.float32)
    all_reward += r
    # print(f"done: {done}")
    done = [done]
    info = [info]
    for i, is_done in enumerate(done):
        if is_done:
            tf.summary.scalar('reward', data=all_reward[i], step=one_episode_reward_index)
            one_episode_reward_index += 1
            all_reward[i] = 0
            if use_RNN:
                total_h[1, i, :] = 0
                total_c[1, i, :] = 0

    # 255+1 loop
    while 1:
        # for epoch in range(1, epochs):
        for epoch in range(1, epochs//size):
            # recv state
            total_state[epoch, :, :, :, :] = recv_state_buf
            remain_step -= 1  # After every recv data minus 1
            if not remain_step:
                break  # leave for loop

            if use_RNN:
                ap, v, total_h[epoch + 1], total_c[epoch + 1] = \
                    model(total_state[epoch], total_h[epoch], total_c[epoch])
            else:
                ap, v = model(recv_state_buf)
            ap = ap.numpy()
            v = v.numpy()
            v.resize((size,))
            total_v[epoch, :] = v
            total_old_ap[epoch, :] = ap

            # scattering action
            a = [np.random.choice(range(a_num), p=ap[i]) for i in range(size)]
            total_a[epoch, :] = a

            # brain's env step
            state_, r, done, info = env.step(a)
            if done:
                state_ = env.reset()
            state = np.array(state_, dtype=np.float32)
            send_state_buf = state

            r = [r]
            done = [done]
            infor = [info]

            total_r[epoch, :] = np.array(r, dtype=np.float32)
            total_done[epoch, :] = np.array(done, dtype=np.float32)
            all_reward += r
            for i, is_done in enumerate(done):
                if is_done:
                    print(i, count_episode[i], all_reward[i])
                    tf.summary.scalar('reward', data=all_reward[i], step=one_episode_reward_index)
                    one_episode_reward_index += 1
                    all_reward[i] = 0
                    count_episode[i] += 1
                    if use_RNN:
                        total_h[epoch + 1, i, :] = 0
                        total_c[epoch + 1, i, :] = 0

        if not remain_step:
            print(rank, 'finished')
            model.save_weights(weight_dir + str(learning_step), save_format='tf')
            break  # leave while 1 loop

        # last one
        remain_step -= 1  # After every recv data minus 1
        # if now remain_step == 0, then exit after last learning

        # only need v
        if use_RNN:
            _, v, _, _ = \
                model(recv_state_buf, total_h[epochs//size], total_c[epochs//size])
                # model(recv_state_buf, total_h[epochs//size], total_c[epochs//size])
        else:
            _, v = model(recv_state_buf)
        v = v.numpy()
        v.resize((size,))
        total_v[-1, :] = v

        #######################
        #    Learning Part    #
        #######################
        # print(f"total state begin learning: {total_state.shape}")
        # critic_v   advantage_v
        total_real_v, total_adv = calc_real_v_and_adv_GAE(total_v, total_r, total_done)
        # total_state.resize((epochs * size, IMG_H, IMG_W, k))
        # total_a.resize((epochs * size,))
        # total_old_ap.resize((epochs * size, a_num))
        # total_adv.resize((epochs * size,))
        # total_real_v.resize((epochs * size,))
        # if use_RNN:
        #     total_h.resize(((epochs + 1) * size, hidden_unit_num))
        #     total_c.resize(((epochs + 1) * size, hidden_unit_num))
        total_state.resize((epochs, IMG_H, IMG_W, k))
        total_a.resize((epochs,))
        total_old_ap.resize((epochs, a_num))
        total_adv.resize((epochs,))
        total_real_v.resize((epochs,))
        if use_RNN:
            total_h.resize(((epochs//size + 1) * size, hidden_unit_num))
            total_c.resize(((epochs//size + 1) * size, hidden_unit_num))

        print('learning' + '-' * 35 + str(learning_step) + '/' + str(total_step // epochs // size))

        # Speed comparison of different methods
        # if learning_step == 100:
        if learning_step == 10:
            import time
            start_time = time.time()
        # if learning_step == 600:
        if learning_step == 30:
            duration = time.time()-start_time
            print(f"duration of 20 learning_step: {duration}")
            print(f"avg duration per learning_step: {duration/20}")
            total_time_counter += time.time()
            print(f"total time duration for 20 learning step: {total_time_counter}")
            # break
            quit()


        # 242.6518578529358
        if use_RNN:
            for _ in range(3):
                # sample_index = np.random.choice(epochs * size, size=epochs * size // 4)
                sample_index = np.random.choice(epochs, size=epochs // 4)
                loss = model.train(total_state[sample_index],
                                   tf.one_hot(total_a, depth=a_num).numpy()[sample_index],
                                   total_adv[sample_index], total_real_v[sample_index],
                                   total_old_ap[sample_index],
                                   total_h[sample_index],
                                   total_c[sample_index])
        else:
            for _ in range(3):
                # sample_index = np.random.choice(epochs * size, size=epochs * size // 4)
                sample_index = np.random.choice(epochs, size=epochs // 4)
                # print(f"total_state: {total_state.shape}")
                # print(f"sample_index: {sample_index.shape}")
                loss = model.train(total_state[sample_index],
                                   tf.one_hot(total_a, depth=a_num).numpy()[sample_index],
                                   total_adv[sample_index], total_real_v[sample_index],
                                   total_old_ap[sample_index])

        # 259.6192126274109
        # np.random.shuffle(total_state)
        # np.random.shuffle(total_a)
        # np.random.shuffle(total_adv)
        # np.random.shuffle(total_real_v)
        # np.random.shuffle(total_old_ap)
        # for i in range(3):
        #     loss = model.train(total_state[batch_size*i:batch_size*(i+1)],
        #                        tf.one_hot(total_a[batch_size*i:batch_size*(i+1)], depth=a_num).numpy(),
        #                        total_adv[batch_size*i:batch_size*(i+1)],
        #                        total_real_v[batch_size*i:batch_size*(i+1)],
        #                        total_old_ap[batch_size*i:batch_size*(i+1)])

        # 322.28919196128845
        # dataset = tf.data.Dataset.from_tensor_slices((total_state, total_a, total_adv, total_real_v, total_old_ap))
        # dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE).shuffle(
        #     buffer_size=epochs * size // 4).batch(
        #     epochs * size // 4)
        # for i, data in enumerate(dataset):
        #     if i == 3:
        #         break
        #     loss = model.train(data[0],
        #                        tf.one_hot(data[1], depth=a_num).numpy(),
        #                        data[2],
        #                        data[3],
        #                        data[4])

        learning_step += 1
        if learning_step % (total_step // epochs // size // 200) == 0:  # recode 200 times
            tf.summary.scalar('loss', data=loss, step=learning_step)
        # if learning_step % (total_step // epochs // size // 3) == 0:  # recode 3 times
        #     model.save_weights(weight_dir + str(learning_step), save_format='tf')

        total_state.resize((epochs, size, IMG_H, IMG_W, k))
        total_a.resize((epochs, size))
        total_old_ap.resize((epochs, size, a_num))
        if use_RNN:
            total_h.resize(((epochs + 1), size, hidden_unit_num))
            total_c.resize(((epochs + 1), size, hidden_unit_num))

        # exit after last learning
        if not remain_step:
            print(rank, 'finished')
            model.save_weights(weight_dir + str(learning_step), save_format='tf')
            break  # leave while 1

        ##############################
        # move last one to first one #
        ##############################

        total_state[0, :, :, :, :] = recv_state_buf
        if use_RNN:
            total_h[0] = total_h[-1]
            total_c[0] = total_c[-1]
        if use_RNN:
            ap, v, total_h[1], total_c[1] = model(total_state[0], total_h[0], total_c[0])
        else:
            ap, v = model(total_state[0])

        ap = ap.numpy()
        v = v.numpy()
        v.resize((size,))
        total_v[0, :] = v
        total_old_ap[0, :] = ap

        # scattering action
        a = [np.random.choice(range(a_num), p=ap[i]) for i in range(size)]
        total_a[0, :] = a

        # brain's env step
        state_, r, done, info = env.step(a)
        if done:
            state_ = env.reset()
        state = np.array(state_, dtype=np.float32)
        send_state_buf = state

        r = [r]
        done = [done]
        infor = [info]

        total_r[0, :] = np.array(r, dtype=np.float32)
        total_done[0, :] = np.array(done, dtype=np.float32)
        all_reward += r
        for i, is_done in enumerate(done):
            if is_done:
                print(i, count_episode[i], all_reward[i])
                tf.summary.scalar('reward', data=all_reward[i], step=one_episode_reward_index)
                one_episode_reward_index += 1
                all_reward[i] = 0
                count_episode[i] += 1
                if use_RNN:
                    total_h[1, i, :] = 0
                    total_c[1, i, :] = 0

        # TODO: test uint8 and float32 who faster
