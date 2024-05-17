from functools import partial

import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit


class Fits:
    def linear(X: npt.ArrayLike, Y: npt.ArrayLike, yerr: npt.ArrayLike | None = None):
        def curve(x, a, b):
            return a * x + b

        if yerr == None:
            param, cov = curve_fit(curve, X, Y, p0=[5, 0])
        else:
            param, cov = curve_fit(curve, X, Y, p0=[5, 0], sigma=yerr)
        std_dev = np.sqrt(np.diag(cov))

        return param, std_dev

    # * Smorzato con C_2 = 0
    @staticmethod
    def curve(x, a, b, c, phi, d):
        return a * np.exp(-b * x) * np.cos(c * x + phi) + d

    def smorzato(X: npt.ArrayLike, Y: npt.ArrayLike):
        param, cov = curve_fit(Fits.curve, X, Y, p0=[100, 0.001, 1, 0, 0])
        std_dev = np.sqrt(np.diag(cov))

        return param, std_dev

    # * Smorzato con C_1 = 0
    @staticmethod
    def curve2(x, a, b, phi, c, /, omega):
        return np.multiply(a / (1 + a * b * x), np.cos(omega * x + phi)) + c

    def smorzato2(X: npt.ArrayLike, Y: npt.ArrayLike, omega: float):
        curve2 = partial(Fits.curve2, omega=omega)
        param, cov = curve_fit(curve2, X, Y, p0=[10, 0.01, 0, 0], maxfev=5000)
        std_dev = np.sqrt(np.diag(cov))

        return param, std_dev

    @staticmethod
    def sinus(x, a, b, c, d):
        return a * np.sin(b * (x - d)) + c

    # ! make possible to give p0 values as argument
    def fit_seno(X, Y):
        param, cov = curve_fit(
            Fits.sinus,
            X,
            Y,
            p0=[10, 7.64, -129, np.pi],
            maxfev=4000,
            bounds=((1, 1, -140, 0), (np.inf, 10, -120, np.pi * 2)),
            method="dogbox",
        )
        std_dev = np.sqrt(np.diag(cov))

        return param, std_dev


class inviluppi:
    pass
