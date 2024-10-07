import sys
import numpy as np


class ChannelModel:
    def __init__(self, APcoord, UEcoord, La, Lb, Ld, fc, d_H, d_V, U):
        self.APlocation = APcoord
        self.UElocation = UEcoord
        self.numberOfPathAPtoRIS = La
        self.numberOfPathUEtoRIS = Lb
        self.numberOfPathAPtoUE = Ld
        self.carrierFrequency = fc
        self.tileLength = d_H
        self.tileHeight = d_V
        self.planarSurface = U

    def reset(self, La, Lb):
       self.numberOfPathAPtoRIS = La
       self.numberOfPathUEtoRIS = Lb

    def channel_samples(self, realizations, bandwidth,
                        subSpacing, numberOfTiles,
                        APtoRIS_LOS, RIStoUE_LOS):

        def sort_all(toSort):
            indexes = np.argsort(-toSort)
            sortedArg = toSort[indexes]

            return sortedArg, indexes

        def spatial_signature(U, varphi, theta, wvlength):
            k = -2 * np.pi / wvlength * \
                np.array([np.cos(varphi) * np.cos(theta),
                          np.sin(varphi) * np.cos(theta),
                          np.sin(theta)]).T

            a = np.exp(1j * np.matmul(k, U))

            return a

        def pathloss_LOS(d):
            pathlossLOS = 10**((-30.18 - 26 * np.log10(d)) / 10)

            return pathlossLOS

        def pathloss_NLOS(d):
            pathlossNLOS = 10**((-34.53 - 38 * np.log10(d)) / 10)

            return pathlossNLOS

        def angle_calc(varphi, angle, numberOfpath):
            LaLb_angles = varphi + \
                np.hstack(
                    [
                        np.array([0]), angle * (
                            2 * np.random.rand(numberOfpath - 1) - 1
                            )
                        ]
                    )

            return LaLb_angles

        def delay_calc(dist, numberOfpath, speed):
            LaLb_delays = dist * (
                1 + np.hstack(
                    [
                        np.array([0]), np.random.rand(numberOfpath - 1)
                        ]
                    )
                ) / speed

            return LaLb_delays
        
        # 25 x 25 m
        def rnd_walk2D(n):
            directions = ["UP", "DOWN", "LEFT", "RIGHT"]
            # Pick a direction at random
            step = [directions[np.random.randint(0, np.size(directions))] for i in range(n)]
            
            for i in range(n):
                # Move the object according to the direction
                if step[i] == "RIGHT":
                    self.UElocation[1] = self.UElocation[1] + (~(self.UElocation[1] == 50) * 2 - 1)
                elif step[i] == "LEFT":
                    self.UElocation[1] = self.UElocation[1] + ((self.UElocation[1] == 1) * 2 - 1)
                elif step[i] == "UP":
                   self.UElocation[0] = self.UElocation[0] + ((self.UElocation[0] == 1) * 2 - 1)
                elif step[i] == "DOWN":
                    self.UElocation[0] = self.UElocation[0] + (~(self.UElocation[0] == 50) * 2 - 1)
                    

        SPEED_OF_LIGHT = 3e8
        wavelength = SPEED_OF_LIGHT / self.carrierFrequency
        AZIMUTH_ANGLE = 40 * np.pi / 180
        ELEVATION_ANGLE = 10 * np.pi / 180

        lossComparedToIsotropic = self.tileLength * self.tileHeight / (wavelength**2 / (4 * np.pi))

        distAP_RIS = np.linalg.norm(self.APlocation)
        varphiAP_RIS = np.arctan(self.APlocation[1]/self.APlocation[0])

        numberOfSubCarriers = np.int_(np.floor(bandwidth / subSpacing))
        V = np.zeros([realizations, numberOfTiles, numberOfSubCarriers], dtype=complex)
        V_LOS = np.zeros([realizations, numberOfTiles, numberOfSubCarriers], dtype=complex)
        hd = np.zeros([realizations, numberOfSubCarriers], dtype=complex)
        M = np.zeros(realizations, dtype=int)

        delayLd = np.random.rand(realizations, self.numberOfPathAPtoUE)
        powfactorLd = np.random.randn(realizations, self.numberOfPathAPtoUE)

        for r in range(realizations):
            sys.stdout.write("\r")
            sys.stdout.write("#%d|%d :: " % (r, realizations))
            sys.stdout.write("Simulating Channel...")
            sys.stdout.flush()

            # self.UElocation = np.hstack([np.random.randint(1, 21, 2), np.zeros(1)])
            # rnd_walk2D(1000)
            distUE_RIS = np.linalg.norm(self.UElocation)
            distAP_UE = np.linalg.norm(self.APlocation - self.UElocation)

            M[r] = np.int_(
                np.round(
                    bandwidth * (2 * (distAP_RIS + distUE_RIS) - distAP_UE) / SPEED_OF_LIGHT
                    ) + 11
                )
            k_iteration = np.array([range(M[r])])

            varphiUE_RIS = np.arctan(self.UElocation[1]/self.UElocation[0])

            La_varphi = angle_calc(varphiAP_RIS, AZIMUTH_ANGLE, self.numberOfPathAPtoRIS)
            Lb_varphi = angle_calc(varphiUE_RIS, AZIMUTH_ANGLE, self.numberOfPathUEtoRIS)
            La_theta = angle_calc(0, ELEVATION_ANGLE, self.numberOfPathAPtoRIS)
            Lb_theta = angle_calc(0, ELEVATION_ANGLE, self.numberOfPathUEtoRIS)

            La_delay = delay_calc(distAP_RIS, self.numberOfPathAPtoRIS, SPEED_OF_LIGHT)
            Lb_delay = delay_calc(distUE_RIS, self.numberOfPathUEtoRIS, SPEED_OF_LIGHT)
            Ld_delay = distAP_UE * (1 + delayLd[r]) / SPEED_OF_LIGHT

            La_powfactor = 10**(-La_delay[1:] + 0.2 * np.random.randn(self.numberOfPathAPtoRIS - 1))
            Lb_powfactor = 10**(-Lb_delay[1:] + 0.2 * np.random.randn(self.numberOfPathUEtoRIS - 1))
            Ld_powfactor = 10**(-Ld_delay  + 0.2 * powfactorLd[r])

            if APtoRIS_LOS:
                RicefactorAP_RIS = 10**((13 - 0.03 * distAP_RIS) / 10)
                La_pathloss = lossComparedToIsotropic * \
                    pathloss_LOS(distAP_RIS) * \
                        np.hstack(
                            [
                                np.array(
                                    [
                                        RicefactorAP_RIS / (1 + RicefactorAP_RIS)
                                        ]
                                    ), (1 / (1 + RicefactorAP_RIS)) * La_powfactor / np.sum(La_powfactor)
                                ]
                            )
            else:
                self.numberOfPathAPtoRIS =  self.numberOfPathAPtoRIS - 1
                La_pathloss, index = sort_all(
                    lossComparedToIsotropic * pathloss_NLOS(distAP_RIS) * La_powfactor / np.sum(La_powfactor)
                    )
                La_varphi = La_varphi[index + 1]
                La_theta = La_theta[index + 1]
                La_delay = La_delay[index + 1]

            if RIStoUE_LOS:
                RicefactorUE_RIS = 10**((13 - 0.03 * distUE_RIS) / 10)
                Lb_pathloss = lossComparedToIsotropic * \
                    pathloss_LOS(distUE_RIS) * \
                        np.hstack(
                            [
                                np.array(
                                    [
                                        RicefactorUE_RIS / (1 + RicefactorUE_RIS)
                                        ]
                                    ), (1 / (1 + RicefactorUE_RIS)) * Lb_powfactor / np.sum(Lb_powfactor)
                                ]
                            )
            else:
                self.numberOfPathUEtoRIS = self.numberOfPathUEtoRIS - 1
                Lb_pathloss,index = sort_all(lossComparedToIsotropic *
                                             pathloss_NLOS(distUE_RIS) *
                                            Lb_powfactor / np.sum(Lb_powfactor))
                Lb_varphi = Lb_varphi[index + 1]
                Lb_theta = Lb_theta[index + 1]
                Lb_delay = Lb_delay[index + 1]

            Ld_pathloss = pathloss_NLOS(distAP_UE) * Ld_powfactor / np.sum(Ld_powfactor)

            eta = np.min(Ld_delay)

            spcSigLa = spatial_signature(self.planarSurface, La_varphi, La_theta, wavelength)
            spcSigLb = spatial_signature(self.planarSurface, Lb_varphi, Lb_theta, wavelength)

            for l1 in range(self.numberOfPathAPtoRIS):
                for l2 in range(self.numberOfPathUEtoRIS):
                    V[r, :, :M[r]] = V[r, :, :M[r]] + np.outer(
                        np.sqrt(La_pathloss[l1] * Lb_pathloss[l2]) *
                        np.exp(-1j * 2 * np.pi * self.carrierFrequency * (La_delay[l1] + Lb_delay[l2])) *
                        spcSigLa[l1] * spcSigLb[l2],
                        np.sinc(k_iteration + bandwidth * (eta - La_delay[l1] - Lb_delay[l2]))
                        )

                    if (l1 == 1) and (l2 == 1):
                        V_LOS[r, :, :M[r]] = V[r, :, :M[r]]

            hd[r, :M[r]] = np.matmul(
                np.sqrt(Ld_pathloss) *
                np.exp(-1j * 2 * np.pi * self.carrierFrequency * Ld_delay),
                np.sinc(k_iteration + bandwidth * (eta - Ld_delay[np.newaxis].T))
                )

        return V, V_LOS, hd, M
