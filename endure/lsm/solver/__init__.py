from typing import Type
from endure.lsm.types import Policy
from .classic_solver import ClassicSolver
from .qlsm_solver import QLSMSolver
from .klsm_solver import KLSMSolver
from .yzlsm_solver import YZLSMSolver


def get_solver(
    choice: Policy,
) -> Type[ClassicSolver | QLSMSolver | KLSMSolver | YZLSMSolver]:
    choices = {
        Policy.Tiering: ClassicSolver,
        Policy.Leveling: ClassicSolver,
        Policy.Classic: ClassicSolver,
        Policy.QFixed: QLSMSolver,
        Policy.YZHybrid: YZLSMSolver,
        Policy.KHybrid: KLSMSolver,
    }
    solver = choices.get(choice, None)
    if solver is None:
        raise KeyError

    return solver
