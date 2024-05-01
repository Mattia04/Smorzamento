import numpy as np
import numpy.typing as npt
from scipy.optimize import curve_fit


class Fits:
    def linear(X: npt.ArrayLike, Y: npt.ArrayLike, yerr: npt.ArrayLike | None):
        def curve(x, a, b):
            return a * x + b

        param, cov = curve_fit(curve, X, Y, p0=[5, 0], sigma=yerr)
        std_dev = np.sqrt(np.diag(cov))

        return param, std_dev

    # * Smorzato con C_2 = 0
    def smorzato(X: npt.ArrayLike, Y: npt.ArrayLike):
        def curve(x, a, b, c, phi, d):
            return a * np.exp(-b * x) * np.cos(c * x + phi) + d

        param, cov = curve_fit(curve, X, Y, p0=[100, 0.001, 1, 0, 0])
        std_dev = np.sqrt(np.diag(cov))

        return param, std_dev

    # * Smorzato con C_1 = 0
    def smorzato2(X: npt.ArrayLike, Y: npt.ArrayLike, omega: float):
        def curve(x, a, b, phi, c):
            # alpha = 4 / 3 * c2 * omega**3 / (np.pi * k) # nel caso da usare per trovare C2
            return np.multiply(a / (1 + a * b * x), np.cos(omega * x + phi)) + c

        param, cov = curve_fit(curve, X, Y, p0=[0.1, 1, 0, 0])
        std_dev = np.sqrt(np.diag(cov))

        return param, std_dev


class inviluppi:
    pass
