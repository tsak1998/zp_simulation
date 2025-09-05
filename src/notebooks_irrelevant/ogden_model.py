from types import LambdaType
from typing import Literal
import jax
import sympy as sp


import sympy as sp


print("r(R) =", sp.simplify(r_expr))
print("Radial stretch (λ_r) =", sp.simplify(lambda_r))
print("Tangential stretch (λ_t) =", sp.simplify(lambda_t))


class OgdenModel:
    """
    This class holds methods that facilitate the fitting of Hyperelastic Ogden Model,
    specifically focusing on Ogden Foam as used in Ansys software, which considers compressible behavior.
    """

    def __init__(self, n: Literal[0, 1, 2], inf_shear: float, inf_bulk: float) -> None:
        pass

    @staticmethod
    def calculate_isochoric_stretches(LAMBDA_1, LAMBDA_2, LAMBDA_3):
        J = LAMBDA_1 * LAMBDA_2 * LAMBDA_3
        J_13 = J ** (-1 / 3)
        return J_13 * LAMBDA_1, J_13 * LAMBDA_2, J_13 * LAMBDA_3

    @staticmethod
    def deviatoric_term(
        LAMBDA_1: float,
        LAMBDA_2: float,
        LAMBDA_3: float,
        alpha: float,
        mi: float,
        J: float,
    ) -> float:
        L1_iso, L2_iso, L3_iso = OgdenModel.calculate_isochoric_stretches(
            LAMBDA_1, LAMBDA_2, LAMBDA_3
        )
        return (mi / alpha) * (
            J ** (alpha / 3) * (L1_iso**alpha + L2_iso**alpha + L3_iso**alpha) - 3
        )

    @staticmethod
    def volumetric_term(alpha: float, mi: float, beta: float, J: float) -> float:
        return (mi / alpha / beta) * (J ** (-alpha * beta) - 1)

    @staticmethod
    def ogden_three_n(
        LAMBDA_1: float,
        LAMBDA_2: float,
        LAMBDA_3: float,
        alpha_1: float,
        mi_1: float,
        beta_1: float,
        alpha_2: float,
        mi_2: float,
        beta_2: float,
        alpha_3: float,
        mi_3: float,
        beta_3: float,
    ) -> float:
        J = LAMBDA_1 * LAMBDA_2 * LAMBDA_3
        deviatoric_term_1 = OgdenModel.deviatoric_term(
            LAMBDA_1, LAMBDA_2, LAMBDA_3, alpha_1, mi_1, J
        )
        deviatoric_term_2 = OgdenModel.deviatoric_term(
            LAMBDA_1, LAMBDA_2, LAMBDA_3, alpha_2, mi_2, J
        )
        deviatoric_term_3 = OgdenModel.deviatoric_term(
            LAMBDA_1, LAMBDA_2, LAMBDA_3, alpha_3, mi_3, J
        )

        volumetric_term_1 = OgdenModel.volumetric_term(alpha_1, mi_1, beta_1, J)
        volumetric_term_2 = OgdenModel.volumetric_term(alpha_2, mi_2, beta_2, J)
        volumetric_term_3 = OgdenModel.volumetric_term(alpha_3, mi_3, beta_3, J)

        W_deviatoric = deviatoric_term_1 + deviatoric_term_2 + deviatoric_term_3
        W_volumetric = volumetric_term_1 + volumetric_term_2 + volumetric_term_3

        return W_deviatoric + W_volumetric

    # The stress functions can remain as defined but using the corrected isochoric stretch calculations

    @staticmethod
    def spherical_radial_stress(
        LAMBDA_1: float,
        LAMBDA_2: float,
        LAMBDA_3: float,
        alpha_1: float,
        mi_1: float,
        beta_1: float,
        alpha_2: float,
        mi_2: float,
        beta_2: float,
        alpha_3: float,
        mi_3: float,
        beta_3: float,
    ) -> float:
        # Total volume ratio
        J = LAMBDA_1 * LAMBDA_2 * LAMBDA_3

        # Compute the derivative of the strain energy with respect to LAMBDA_1.
        # Note: We pass the original lambda values (without scaling) because
        # the energy function itself calculates the isochoric stretches.
        dW_dLAMBDA_1 = jax.grad(
            lambda x: OgdenModel.ogden_three_n(
                x,
                LAMBDA_2,
                LAMBDA_3,
                alpha_1,
                mi_1,
                beta_1,
                alpha_2,
                mi_2,
                beta_2,
                alpha_3,
                mi_3,
                beta_3,
            )
        )(LAMBDA_1)

        # Cauchy (true) stress: sigma_rr = (LAMBDA_1 / J) * dW/dLAMBDA_1
        return (LAMBDA_1 / J) * dW_dLAMBDA_1

    @staticmethod
    def spherical_tang_stress(
        LAMBDA_1: float,
        LAMBDA_2: float,
        LAMBDA_3: float,
        alpha_1: float,
        mi_1: float,
        beta_1: float,
        alpha_2: float,
        mi_2: float,
        beta_2: float,
        alpha_3: float,
        mi_3: float,
        beta_3: float,
    ) -> float:
        # Total volume ratio
        J = LAMBDA_1 * LAMBDA_2 * LAMBDA_3

        # Compute the derivative of the strain energy with respect to LAMBDA_2.
        dW_dLAMBDA_2 = jax.grad(
            lambda x: OgdenModel.ogden_three_n(
                LAMBDA_1,
                x,
                LAMBDA_3,
                alpha_1,
                mi_1,
                beta_1,
                alpha_2,
                mi_2,
                beta_2,
                alpha_3,
                mi_3,
                beta_3,
            )
        )(LAMBDA_2)

        # Cauchy stress: sigma_tt = (LAMBDA_2 / J) * dW/dLAMBDA_2
        return (LAMBDA_2 / J) * dW_dLAMBDA_2


class GeneralCalculator:

    @staticmethod
    def calculate_principal_stretch(L0: float, L: float) -> float:
        return L0 + L

    @staticmethod
    def calculate_jacobian(
        LAMBDA_1: float, LAMBDA_2: float, LAMBDA_3: float
    ) -> float: ...

    @staticmethod
    def a(): ...
