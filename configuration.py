import sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cvxpy as cp
import numpy as np

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

class strongestTapMax:
    def config_samples(channelAPtoUE, strongestPath, isQtz, QtzStp):
        indexes = np.argmax(
            np.abs(channelAPtoUE) + np.sum(
                np.abs(strongestPath), axis=1
                ), axis=1
            )
        config_rotations = np.angle(
            channelAPtoUE[np.arange(indexes.size), indexes]
            )[:, np.newaxis] -\
            np.angle(strongestPath[np.arange(indexes.size), :, indexes])

        if isQtz:
            quantizerStep = QtzStp
            config_rotations = np.exp(1j * (
                np.round(config_rotations / quantizerStep) * quantizerStep
                )
                )
        else:
            config_rotations = np.exp(1j * config_rotations)

        return config_rotations

class NeuralNet:
    def __init__(self, N, K, isQtz, QtzStp):
        self.numberOfTiles = N
        self.numberOfSubcarriers = K
        self.NeuralNetModel = []
        self.isTrained = False
        self.isQuantized = isQtz
        self.QuantizerStep = QtzStp
        
    def soft_round(x, alpha=7., eps=10**-3):
        # This guards the gradient of tf.where below against NaNs, while maintaining
        # correctness, as for alpha < eps the result is ignored.
        alpha_bounded = tf.maximum(alpha, eps)

        m = tf.floor(x) + .5
        r = x - m
        z = tf.tanh(alpha_bounded / 2.) * 2.
        y = m + tf.tanh(alpha_bounded * r) / z

        # For very low alphas, soft_round behaves like identity
        return tf.where(alpha < eps, x, y, name="soft_round")

    def mult_phasor(n1, w2):
        a1, w1 = tf.abs(n1), tf.math.angle(n1)

        real_part = a1 * tf.math.cos(w1 + tf.cast(w2, tf.float64))
        imag_part = a1 * tf.math.sin(w1 + tf.cast(w2, tf.float64))

        return tf.complex(real_part, imag_part)

    def build_NeuralNet(self):
        class InputDense(layers.Layer):
            def __init__(self, units=32):
                super(InputDense, self).__init__()
                self.units = units

            def build(self, input_shape):
                self.w = self.add_weight(
                    shape=(input_shape[-1], self.units // 1),
                    initializer=keras.initializers.GlorotUniform(seed=42),
                    trainable=True,)

                self.b = self.add_weight(
                    shape=(self.units,),
                    initializer=keras.initializers.GlorotUniform(seed=42),
                    trainable=True)

            def call(self, inputs_1, inputs_2):
                phase_dealignment = tf.expand_dims(tf.math.angle(inputs_1), axis=1) - tf.math.angle(inputs_2)
                return tf.nn.relu(
                    tf.math.reduce_sum(
                        tf.matmul(phase_dealignment, self.w), axis=2
                        )
                    ) + self.b

            def get_config(self):
                return {"units": self.units}

        class HiddenDense(layers.Layer):
            def __init__(self, units=32):
                super(HiddenDense, self).__init__()
                self.units = units

            def build(self, input_shape):
                self.w = self.add_weight(
                    shape=(input_shape[-1],),
                    initializer=keras.initializers.GlorotUniform(seed=42),
                    trainable=True,)

                self.b = self.add_weight(
                    shape=(self.units,),
                    initializer=keras.initializers.GlorotUniform(seed=42),
                    trainable=True)

            def call(self, inputs):
                return tf.nn.relu(inputs * self.w) + self.b

            def get_config(self):
                return {"units": self.units}

        inputs_1 = keras.Input((self.numberOfSubcarriers,))
        inputs_2 = keras.Input((self.numberOfTiles, self.numberOfSubcarriers))
        x1 = InputDense(self.numberOfTiles)(inputs_1, inputs_2)
        y1 = HiddenDense(self.numberOfTiles)(x1)
        y2 = HiddenDense(self.numberOfTiles)(y1)
        y3 = HiddenDense(self.numberOfTiles)(y2)
        y4 = HiddenDense(self.numberOfTiles)(y3)

        self.NeuralNetModel = keras.Model(inputs=[inputs_1, inputs_2], outputs=y4)

    def train_NeuralNet(self, MAX_EPOCHS, ITERATIONS_RUNS, compositeChannel, channelAPtoUE, strongestPath):
        def loss_fn(predicted_y, qtzStp, F):
            if self.isQuantized: predicted_y = NeuralNet.soft_round(predicted_y / qtzStp) * qtzStp
            rotatedChannel = tf.math.reduce_sum(
                NeuralNet.mult_phasor(
                    compositeChannel, tf.expand_dims(predicted_y, axis=2)
                    ), axis=1
                )
            combined_frequencyDomain = tf.matmul(
                tf.expand_dims(
                    channelAPtoUE + rotatedChannel, axis=1
                    ), F, adjoint_b=True
                )
            return tf.math.reduce_sum(
                -tf.square(
                    tf.abs(
                        tf.squeeze(combined_frequencyDomain)
                        )
                    )
                ) / (predicted_y.shape[0] + self.numberOfSubcarriers)

        INIT_ETA = 10**-2
        TOL = 10**-2
        N_ITER_NO_CHANGE = 10
        NO_IMP_COUNT = 0
        best_loss = np.inf
        quantizerStep = tf.constant(self.QuantizerStep)

        NeuralNet.build_NeuralNet(self)

        F = tf.signal.fft(tf.eye(self.numberOfSubcarriers, dtype=tf.complex128))
        APtoUE_frequencyDomain = tf.squeeze(
            tf.matmul(
                tf.expand_dims(channelAPtoUE, axis=1), F, adjoint_b=True
                )
            )
        strongestPath_frequencyDomain = tf.squeeze(
            tf.matmul(
                tf.expand_dims(strongestPath, axis=1), F, adjoint_b=True
                )
            )

        optimizer = keras.optimizers.Adam(learning_rate=INIT_ETA)
        self.isTrained = True
        try:
            for epoch in range(MAX_EPOCHS):
                # print("\nEpoch #%d" % (epoch,))
                sys.stdout.write("\r")
                sys.stdout.write("#%d|%d :: " % (epoch, MAX_EPOCHS))
                sys.stdout.write("Training..." + 11 * " ")
                sys.stdout.flush()

                for step in range(ITERATIONS_RUNS):
                    with tf.GradientTape() as tape:
                        out = self.NeuralNetModel(
                            (APtoUE_frequencyDomain, strongestPath_frequencyDomain),
                            training=True
                            )
                        loss_value = loss_fn(out, quantizerStep, F) * 10**9

                    grads = tape.gradient(loss_value, self.NeuralNetModel.trainable_weights)
                    optimizer.apply_gradients(zip(grads, self.NeuralNetModel.trainable_weights))

                    # sys.stdout.write('\r')
                    # sys.stdout.write("%.1f" % (100 / ITERATIONS_RUNS * (step + 1)))
                    # sys.stdout.flush()
                
                # sys.stdout.write("\n Loss: {:.4f}".format(loss_value))
                # sys.stdout.write("\n Test Loss: {:.4f}".format(loss_valueTest))

                if loss_value > best_loss - TOL:
                    NO_IMP_COUNT += 1
                    if NO_IMP_COUNT >= N_ITER_NO_CHANGE:
                        # sys.stdout.write("\n Loss: {:.4f}".format(loss_value))
                        sys.stdout.flush()
                        break
                else:
                    NO_IMP_COUNT = 0
                if loss_value < best_loss:
                    best_loss = loss_value

                # sys.stdout.write("\n No improvement count: {:.1f}".format(NO_IMP_COUNT))
                # sys.stdout.write("\n Best Loss: {:.4f}".format(best_loss))
                # sys.stdout.flush()

        except KeyboardInterrupt:
            NO_IMP_COUNT = np.inf

    def feedFoward_NeuralNet(self, channelAPtoUE, strongestPath):
        if self.isTrained:
            F = tf.signal.fft(tf.eye(self.numberOfSubcarriers, dtype=tf.complex128))
            APtoUE_frequencyDomain = tf.squeeze(
                tf.matmul(
                    tf.expand_dims(channelAPtoUE, axis=1), F, adjoint_b=True
                    )
                )
            strongestPath_frequencyDomain = tf.squeeze(
                tf.matmul(
                    tf.expand_dims(strongestPath, axis=1), F, adjoint_b=True
                    )
                )
            config_rotations = self.NeuralNetModel(
                (APtoUE_frequencyDomain, strongestPath_frequencyDomain),
                training=False
                )

            if self.isQuantized:
                quantizerStep = self.QuantizerStep
                config_rotations = np.exp(1j * (
                    np.round(config_rotations.numpy() / quantizerStep) * quantizerStep
                    )
                    )
            else:
                config_rotations = np.exp(1j * config_rotations.numpy())
        else:
            raise RuntimeError("Neural network must be trained first!")

        return config_rotations

class ConvexSolver:
    def __init__(self, N, K, B, isQtz, QtzStp):
        self.numberOfTiles = N
        self.numberOfSubcarriers = K
        self.bandwidth = B
        self.isQuantized = isQtz
        self.QuantizerStep = QtzStp

    def alternate_opt(self, hd, V, M):
        def water_filling(totalPower,lambdaInv):
            Nt = np.size(lambdaInv)
            lambdaInvSorted = np.sort(lambdaInv)

            alpha_candidates = (totalPower + np.cumsum(lambdaInvSorted)) / (np.array(range(Nt)) + 1)
            optimalIndex = (alpha_candidates - lambdaInvSorted > 0) * \
                (alpha_candidates - np.hstack([lambdaInvSorted[1:], np.array([np.inf])]) < 0)
            waterLevel = alpha_candidates[optimalIndex]

            powerAllocation = waterLevel - lambdaInv
            powerAllocation[powerAllocation < 0] = 0

            return powerAllocation

        def convex_problem(K, N, waterFillPower, phiTilde, gamma, hd, V, F):
            p = cp.Parameter(K); p = waterFillPower
            phi = cp.Variable(N, complex=True)
            y = cp.Variable(K); a = cp.Variable(K); b = cp.Variable(K);
            logConst = cp.Constant(np.log(2))

            aTilde = np.real(F.conj() @ (hd + V.T @ phiTilde))
            bTilde = np.imag(F.conj() @ (hd + V.T @ phiTilde))

            constraints = [cp.abs(phi) <= 1]
            constraints += [a == cp.real(F.conj() @ (hd + V.T @ phi))]
            constraints += [b == cp.imag(F.conj() @ (hd + V.T @ phi))]
            constraints += [y <= cp.square(aTilde) + cp.square(bTilde) +
                            cp.multiply(2 * aTilde, a - aTilde) +
                            cp.multiply(2 * bTilde, b - bTilde)
                            ]
            prob = cp.Problem(
                cp.Maximize(
                    cp.sum(cp.log(1 + cp.multiply(y, p)) / logConst)
                    ), constraints)

            prevValue = np.inf; TOL = 10**-2
            while True:
                prob.solve(solver=cp.SCS,verbose=False)
                phi_out = np.array(phi.value) / np.abs(np.array(phi.value))
                # phi_out = np.array(phi.value)

                convCriteria = np.abs(np.sum(np.angle(phi_out) - prevValue))

                # sys.stdout.write("\r")
                # sys.stdout.write("\n    Inner Convergence: {:.4f}".format(convCriteria))
                # sys.stdout.flush()

                if convCriteria > TOL:
                    prevValue = np.angle(phi_out)
                    aTilde = np.array(a.value)
                    bTilde = np.array(b.value)
                else:
                    break

            return phi_out

        power = 1 * self.bandwidth / 1e6
        noisePower = 10**((-174 + 10) / 10) / 1000
        F = np.fft.fft(np.eye(self.numberOfSubcarriers))

        runs = hd.shape[0]; TOL = 10**-1
        # saIterations = 10
        rate = np.zeros([runs, self.numberOfSubcarriers])
        config_rotations = np.zeros([runs, self.numberOfTiles], dtype=complex)
        config_pwrAllocations = np.zeros([runs, self.numberOfSubcarriers])
        for r in range(runs):
            # sys.stdout.write("\n")
            sys.stdout.write("\r")
            sys.stdout.write("#%d|%d :: " % (r, runs))
            sys.stdout.write("Running CVX Solver...")
            sys.stdout.flush()

            initPhi = np.pi * (2 * np.random.rand(self.numberOfTiles) - 1)

            initPhi = [
                np.exp(-1j *
                       (
                           (
                               np.conj(hd[r]) +
                               np.conj(V[r, np.arange(self.numberOfTiles) != i]).T @
                               initPhi[np.arange(self.numberOfTiles) != i]
                               ) @ V[r, i]
                           )
                       ) for i in range(self.numberOfTiles)
                ]
            initPhi = np.array(initPhi)

            prevRate = np.inf
            while True:
                combined = F.conj() @ (hd[r] + V[r].T @ initPhi)
                SNR = power / (self.bandwidth * noisePower) * np.abs(combined)**2

                powerAllocation = water_filling(self.numberOfSubcarriers, 1 / SNR)

                rate[r] = self.bandwidth / (self.numberOfSubcarriers + M[r] - 1) * \
                    np.log2(1 + (SNR * powerAllocation))

                convCriteriaOuter = np.abs(np.sum(rate[r]) - prevRate) / self.bandwidth

                # sys.stdout.write("\r")
                # sys.stdout.write("\n\n Outer Convergence: {:.4f}".format(np.sum(rate[r] / self.bandwidth)))
                # sys.stdout.flush()

                if convCriteriaOuter > TOL:
                    prevRate = np.sum(rate[r])
                else:
                    config_rotations[r] = np.angle(initPhi)
                    config_pwrAllocations[r] = powerAllocation
                    break

                initPhi = convex_problem(self.numberOfSubcarriers, self.numberOfTiles,
                                         powerAllocation, initPhi,
                                         noisePower, hd[r], V[r], F)
        
        if self.isQuantized:
            quantizerStep = self.QuantizerStep
            config_rotations = np.exp(1j * (
                np.round(config_rotations / quantizerStep) * quantizerStep
                )
                )
        else:
            config_rotations = np.exp(1j * config_rotations)
        
        return config_rotations, config_pwrAllocations
