from keras import backend as K
from keras import regularizers
from keras.layers import Dense, Dropout
from keras.models import Sequential


def significanceLoss(expectedSignal, expectedBkgd):
    """Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size"""

    def sigLoss(y_true, y_pred):
        # Continuous version:

        signalWeight = expectedSignal / K.sum(y_true)
        bkgdWeight = expectedBkgd / K.sum(1 - y_true)

        s = signalWeight * K.sum(y_pred * y_true)
        b = bkgdWeight * K.sum(y_pred * (1 - y_true))

        return -(s * s) / (
            s + b + K.epsilon()
        )  # Add the epsilon to avoid dividing by 0

    return sigLoss


def significanceLossInvert(expectedSignal, expectedBkgd):
    """Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size"""

    def sigLossInvert(y_true, y_pred):
        # Continuous version:

        signalWeight = expectedSignal / K.sum(y_true)
        bkgdWeight = expectedBkgd / K.sum(1 - y_true)

        s = signalWeight * K.sum(y_pred * y_true)
        b = bkgdWeight * K.sum(y_pred * (1 - y_true))

        return (s + b) / (s * s + K.epsilon())  # Add the epsilon to avoid dividing by 0

    return sigLossInvert


def significanceLoss2Invert(expectedSignal, expectedBkgd):
    """Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size"""

    def sigLoss2Invert(y_true, y_pred):
        # Continuous version:

        signalWeight = expectedSignal / K.sum(y_true)
        bkgdWeight = expectedBkgd / K.sum(1 - y_true)

        s = signalWeight * K.sum(y_pred * y_true)
        b = bkgdWeight * K.sum(y_pred * (1 - y_true))

        return b / (s * s + K.epsilon())  # Add the epsilon to avoid dividing by 0

    return sigLoss2Invert


def significanceLossInvertSqrt(expectedSignal, expectedBkgd):
    """Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size"""

    def sigLossInvert(y_true, y_pred):
        # Continuous version:

        signalWeight = expectedSignal / K.sum(y_true)
        bkgdWeight = expectedBkgd / K.sum(1 - y_true)

        s = signalWeight * K.sum(y_pred * y_true)
        b = bkgdWeight * K.sum(y_pred * (1 - y_true))

        return K.sqrt(s + b) / (
            s + K.epsilon()
        )  # Add the epsilon to avoid dividing by 0

    return sigLossInvert


def significanceFull(expectedSignal, expectedBkgd):
    """Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size"""

    def significance(y_true, y_pred):
        # Discrete version

        signalWeight = expectedSignal / K.sum(y_true)
        bkgdWeight = expectedBkgd / K.sum(1 - y_true)

        s = signalWeight * K.sum(K.round(y_pred) * y_true)
        b = bkgdWeight * K.sum(K.round(y_pred) * (1 - y_true))

        return s / K.sqrt(s + b + K.epsilon())  # Add the epsilon to avoid dividing by 0

    return significance


def asimovSignificanceLoss(expectedSignal, expectedBkgd, systematic):
    """Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size"""

    def asimovSigLoss(y_true, y_pred):
        # Continuous version:

        signalWeight = expectedSignal / K.sum(y_true)
        bkgdWeight = expectedBkgd / K.sum(1 - y_true)

        s = signalWeight * K.sum(y_pred * y_true)
        b = bkgdWeight * K.sum(y_pred * (1 - y_true))
        sigB = systematic * b

        return -2 * (
            (s + b)
            * K.log(
                (s + b)
                * (b + sigB * sigB)
                / (b * b + (s + b) * sigB * sigB + K.epsilon())
                + K.epsilon()
            )
            - b
            * b
            * K.log(1 + sigB * sigB * s / (b * (b + sigB * sigB) + K.epsilon()))
            / (sigB * sigB + K.epsilon())
        )  # Add the epsilon to avoid dividing by 0

    return asimovSigLoss


def asimovSignificanceLossInvert(expectedSignal, expectedBkgd, systematic):
    """Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size"""

    def asimovSigLossInvert(y_true, y_pred):
        # Continuous version:

        signalWeight = expectedSignal / K.sum(y_true)
        bkgdWeight = expectedBkgd / K.sum(1 - y_true)

        s = signalWeight * K.sum(y_pred * y_true)
        b = bkgdWeight * K.sum(y_pred * (1 - y_true))
        sigB = systematic * b

        return 1.0 / (
            2
            * (
                (s + b)
                * K.log(
                    (s + b)
                    * (b + sigB * sigB)
                    / (b * b + (s + b) * sigB * sigB + K.epsilon())
                    + K.epsilon()
                )
                - b
                * b
                * K.log(1 + sigB * sigB * s / (b * (b + sigB * sigB) + K.epsilon()))
                / (sigB * sigB + K.epsilon())
            )
        )  # Add the epsilon to avoid dividing by 0

    return asimovSigLossInvert


def asimovSignificanceFull(expectedSignal, expectedBkgd, systematic):
    """Define a loss function that calculates the significance based on fixed
    expected signal and expected background yields for a given batch size"""

    def asimovSignificance(y_true, y_pred):
        # Continuous version:

        signalWeight = expectedSignal / K.sum(y_true)
        bkgdWeight = expectedBkgd / K.sum(1 - y_true)

        s = signalWeight * K.sum(K.round(y_pred) * y_true)
        b = bkgdWeight * K.sum(K.round(y_pred) * (1 - y_true))
        sigB = systematic * b

        return K.sqrt(
            2
            * (
                (s + b)
                * K.log(
                    (s + b)
                    * (b + sigB * sigB)
                    / (b * b + (s + b) * sigB * sigB + K.epsilon())
                    + K.epsilon()
                )
                - b
                * b
                * K.log(1 + sigB * sigB * s / (b * (b + sigB * sigB) + K.epsilon()))
                / (sigB * sigB + K.epsilon())
            )
        )  # Add the epsilon to avoid dividing by 0

    return asimovSignificance


def truePositive(y_true, y_pred):
    return K.sum(K.round(y_pred) * y_true) / (K.sum(y_true) + K.epsilon())


def falsePositive(y_true, y_pred):
    return K.sum(K.round(y_pred) * (1 - y_true)) / (K.sum(1 - y_true) + K.epsilon())
