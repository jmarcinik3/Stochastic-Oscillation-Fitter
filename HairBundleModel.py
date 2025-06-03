from numbers import Number
from typing import Iterable, TypeVar
import sympy as sym
from sympy import Mod, Piecewise, Symbol, exp, sin

from Fitter import SdeModelSymbolic


number_or_symbol = TypeVar("Parameter", Number, Symbol)


def periodicForce(
    t: number_or_symbol = Symbol("t"),
    A: number_or_symbol = Symbol("A"),
    f: number_or_symbol = Symbol("f"),
    phi0: number_or_symbol = 0,
    F0: number_or_symbol = 0,
    shape: str = "sine",
):
    match shape:
        case "sine":
            core_force = sin(2 * sym.pi * f * t + phi0)
        case "triangle":
            phi = phi0 / (2 * sym.pi) + 0.25
            proportion = Mod(f * t + phi, 1)
            core_force = Piecewise(
                (4 * proportion - 1, proportion < 0.5),
                (-4 * (proportion - 1) - 1, proportion >= 0.5),
            )
        case _:
            message = f"Shape must be sine, triangle, or square {shape}"
            raise ValueError(message)

    return sym.simplify(A * core_force + F0)


class Nondimensional(SdeModelSymbolic):
    @staticmethod
    def kgs(
        pgs: Symbol,
        kgsmin: number_or_symbol,
    ) -> number_or_symbol:
        return 1 - pgs * (1 - kgsmin)

    @staticmethod
    def xgs(
        xhb: Symbol,
        xa: Symbol,
        chihb: number_or_symbol,
        chia: number_or_symbol,
        xc: number_or_symbol,
    ) -> number_or_symbol:
        return chihb * xhb - chia * xa + xc

    @staticmethod
    def Fgs(
        pT: Symbol,
        kgs: number_or_symbol,
        xgs: number_or_symbol,
    ) -> number_or_symbol:
        return kgs * (xgs - pT)

    @staticmethod
    def C(
        pm: Symbol,
        Cmin: number_or_symbol,
    ) -> number_or_symbol:
        return 1 - pm * (1 - Cmin)

    @staticmethod
    def S(
        pm: Symbol,
        Smin: number_or_symbol,
    ) -> number_or_symbol:
        return Smin + pm * (1 - Smin)

    @staticmethod
    def pT_inf(
        kgs: number_or_symbol,
        xgs: number_or_symbol,
        Ugsmax: number_or_symbol,
        dE0: number_or_symbol,
    ) -> number_or_symbol:
        return 1 / (1 + exp(Ugsmax * (dE0 - kgs * (xgs - 1 / 2))))

    @staticmethod
    def xhb_dot(
        xhb: Symbol,
        Fgs: number_or_symbol,
        tauhb0: number_or_symbol,
    ) -> number_or_symbol:
        return -(Fgs + xhb) / tauhb0

    @staticmethod
    def xa_dot(
        xa: Symbol,
        Fgs: number_or_symbol,
        C: number_or_symbol,
        S: number_or_symbol,
        Smax: number_or_symbol,
    ) -> number_or_symbol:
        Cmax = 1 - Smax
        return Smax * S * (Fgs - xa) - Cmax * C

    @staticmethod
    def pm_dot(
        pm: Symbol,
        pT: Symbol,
        taum0: number_or_symbol,
        Cm: number_or_symbol,
    ) -> number_or_symbol:
        return (Cm * pT * (1 - pm) - pm) / taum0

    @staticmethod
    def pgs_dot(
        pT: Symbol,
        pgs: Symbol,
        taugs0: number_or_symbol,
        Cgs: number_or_symbol,
    ) -> number_or_symbol:
        return (Cgs * pT * (1 - pgs) - pgs) / taugs0

    @staticmethod
    def pT_dot(
        pT: Symbol,
        pT_inf: number_or_symbol,
        tauT0: number_or_symbol,
    ) -> number_or_symbol:
        return (pT_inf - pT) / tauT0

    def __init__(
        self,
        Cmin: number_or_symbol = Symbol("Cmin"),
        tauT0: number_or_symbol = Symbol("tauT0"),
        taugs0: number_or_symbol = Symbol("taugs0"),
        Cm: number_or_symbol = Symbol("Cm"),
        xc: number_or_symbol = Symbol("xc"),
        chihb: number_or_symbol = Symbol("chihb"),
        Smin: number_or_symbol = Symbol("Smin"),
        Cgs: number_or_symbol = Symbol("Cgs"),
        tauhb0: number_or_symbol = Symbol("tauhb0"),
        Smax: number_or_symbol = Symbol("Smax"),
        Ugsmax: number_or_symbol = Symbol("Ugsmax"),
        chia: number_or_symbol = Symbol("chia"),
        kgsmin: number_or_symbol = Symbol("kgsmin"),
        taum0: number_or_symbol = Symbol("taum0"),
        dE0: number_or_symbol = Symbol("dE0"),
        parameter_names: Iterable[str] = None,
    ):
        xhb = Symbol("xhb")
        xa = Symbol("xa")
        pm = Symbol("pm")
        pgs = Symbol("pgs")
        pT = Symbol("pT")
        variable_symbols = [xhb, xa, pm, pgs, pT]

        parameters = (
            Cmin,
            tauT0,
            taugs0,
            Cm,
            xc,
            chihb,
            Smin,
            Cgs,
            tauhb0,
            Smax,
            Ugsmax,
            chia,
            kgsmin,
            taum0,
            dE0,
        )
        self._assertParametersType(parameters)
        parameter_symbols = [
            parameter for parameter in parameters if isinstance(parameter, Symbol)
        ]

        kgs = self.kgs(pgs, kgsmin)
        xgs = self.xgs(xhb, xa, chihb, chia, xc)
        Fgs = self.Fgs(pT, kgs, xgs)
        C = self.C(pm, Cmin)
        S = self.S(pm, Smin)
        pT_inf = self.pT_inf(kgs, xgs, Ugsmax, dE0)

        xhb_dot: sym.Expr = sym.simplify(self.xhb_dot(xhb, Fgs, tauhb0))
        xa_dot: sym.Expr = sym.simplify(self.xa_dot(xa, Fgs, C, S, Smax))
        pm_dot: sym.Expr = sym.simplify(self.pm_dot(pm, pT, taum0, Cm))
        pgs_dot: sym.Expr = sym.simplify(self.pgs_dot(pT, pgs, taugs0, Cgs))
        pT_dot: sym.Expr = sym.simplify(self.pT_dot(pT, pT_inf, tauT0))
        derivative_expressions = sym.Tuple(xhb_dot, xa_dot, pm_dot, pgs_dot, pT_dot)

        SdeModelSymbolic.__init__(
            self,
            derivative_expressions,
            variable_symbols,
            parameter_symbols,
        )
        self.reorderParameters(parameter_names)

        if tauT0 == 0:
            self.assumeEquilibrium("pT")
        elif taugs0 == 0:
            self.assumeEquilibrium("pgs")
        elif tauhb0 == 0:
            self.assumeEquilibrium("xhb")
        elif taum0 == 0:
            self.assumeEquilibrium("pm")


class NondimensionalWithForce(Nondimensional):
    @staticmethod
    def F_stim(
        t: Symbol,
        A: number_or_symbol,
        f: number_or_symbol,
        F0: number_or_symbol,
        phi0: number_or_symbol,
        shape: str,
    ):
        return periodicForce(
            t=t,
            A=A,
            f=f,
            phi0=phi0,
            F0=F0,
            shape=shape,
        )

    def __init__(
        self,
        tauhb0: number_or_symbol = Symbol("tauhb0"),
        Famp: number_or_symbol = Symbol("Famp"),
        Ffreq: number_or_symbol = Symbol("Ffreq"),
        force_shape: str = "sine",
        F0: number_or_symbol = 0,
        Fphi: number_or_symbol = 0,
        **kwargs,
    ):
        assert tauhb0 != 0

        Nondimensional.__init__(
            self,
            tauhb0=tauhb0,
            **kwargs,
        )
        force = self.F_stim(
            t=self.time,
            A=Famp,
            f=Ffreq,
            F0=F0,
            phi0=Fphi,
            shape=force_shape,
        )
        self.addTermToDerivative("xhb", force)
